import os
import re
import glob
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset

# import wandb
import evaluate
import transformers
from transformers import Trainer, Seq2SeqTrainingArguments
# from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

class AudioConverter:
    """
    AudioConverter offers methods to load, transcode and augment
    audio data in various ways.
    """
    resampleFilterParams = {
        "fast": {  # Fast and less accurate but still MSE = ~2e-5 compared to librosa.
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "resampling_method": "kaiser_window",
            "beta": 8.555504641634386,
        },
        "best": { # Twice as slow, and a little bit more accburate.
            "lowpass_filter_width": 64,
            "rolloff": 0.9475937167399596,
            "resampling_method": "kaiser_window",
            "beta": 14.769656459379492,       
        },
    }

    def __init__(
        self,
        sampleRate: int,
        disableAug: bool = False,
        speedAugProb: float = 0.7,
        volAugProb: float = 0.7,
        reverbAugProb: float = 0.40,
        noiseAugProb: float = 0.7,
        downsample_8KProb : float = 0.30,
        speedFactors: Tuple[float, float] = None,
        volScaleMinMax: Tuple[float, float] = None,
        reverbRoomScaleMinMax: Tuple[float, float] = None,
        reverbHFDampingMinMax: Tuple[float, float] = None,
        reverbSustainMinMax: Tuple[float, float] = None,
        noiseSNRMinMax: Tuple[float, float] = None,
        noiseFileList: List[str] = None,
    ):
        """
        Initializes AudioConverter.

        Parameters
        ----------
        sampleRate: int
            Sampling rate to convert audio to, if required.

        disableAug: bool, optional
            If True, overrides all other augmentation CONFIGs and
            disables all augmentatoins.

        speedAugProb: float, optional
            Probability that speed augmentation will be applied.
            If <= 0, speed augmentation is disabled.

        volAugProb: float, optional
            Probability that volume augmentation will be applied.
            If <= 0, volume augmentation is disabled.

        reverbAugProb: float, optional
            Probability that reverberation augmentation will be applied.
            If <= 0, reverberation augmentation is disabled.

        noiseAugProb: float, optional
            Probability that noise augmentation will be applied.
            If <= 0, noise augmentation is disabled.

        speedFactors: List[float], optional
            List of factors by which to speed up (>1) or slow down (<1)
            audio by. One factor is chosen randomly if provided. Otherwise,
            default speed factors are [0.9, 1.0, 1.0].
            
        volScaleMinMax: Tuple[float, float], optional
            [Min, Max] range for volume scale factors. One factor is
            chose randomly with uniform probability from this range.
            Default range is [0.125, 2.0].

        reverbRoomScaleMinMax: Tuple[float, float], optional
            [Min, Max] range for room size percentage. Values must be
            between 0 and 100. Larger room size results in more reverb.
            Default range is [25, 75].

        reverbHFDampingMinMax: Tuple[float, float], optional
            [Min, Max] range for high frequency damping percentage. Values must
            be between 0 and 100. More damping results in muffled sound.
            Default range is [25, 75].
        
        reverbSustainMinMax: Tuple[float, float], optional
            [Min, Max] range for reverberation sustain percentage. Values must
            be between 0 and 100. More sustain results in longer lasting echoes.
            Default range is [25, 75].
            
        noiseSNRMinMax: Tuple[float, float], optional
            [Min, Max] range for signal-to-noise ratio when adding noise. One
            factor is chose randomly with uniform probability from this range.
            Lower SNR results in louder noise. Default range is [10.0, 30.0].

        noiseFileList: List[str], optional
            List of paths to audio files to use as noise samples. If None is provided,
            noise augmentation will be disabled. Otherwise, the audio files will be assumed
            to be sources of noise, and be mixed in with speech audio on-the-fly.
        """
        self.sampleRate = sampleRate
        
        enableAug = not disableAug
        self.speedAugProb = speedAugProb if enableAug else -1
        self.volAugProb = volAugProb if enableAug else -1
        self.reverbAugProb = reverbAugProb if enableAug else -1
        self.noiseAugProb = noiseAugProb if enableAug else -1
        self.downsample_8KProb = downsample_8KProb if enableAug else -1
        self.resampleFilterParams = {
                                        "fast": {  # Fast and less accurate but still MSE = ~2e-5 compared to librosa.
                                            "lowpass_filter_width": 16,
                                            "rolloff": 0.85,
                                            "resampling_method": "kaiser_window",
                                            "beta": 8.555504641634386,
                                        },
                                        "best": { # Twice as slow, and a little bit more accburate.
                                            "lowpass_filter_width": 64,
                                            "rolloff": 0.9475937167399596,
                                            "resampling_method": "kaiser_window",
                                            "beta": 14.769656459379492,       
                                        },
                                    }
        
        # Factors by which audio speed is perturbed.
        self.speedFactors = speedFactors
        if speedFactors is None:
            self.speedFactors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        # [Min, Max] Volume scale range.
        self.volScaleRange = volScaleMinMax
        if volScaleMinMax is None:
            self.volScaleRange = [0.125, 2.0]
        
        # [Min, Max] Room size as a percentage, higher = more reverb
        self.reverbRoomScaleRange = reverbRoomScaleMinMax
        if reverbRoomScaleMinMax is None:
            self.reverbRoomScaleRange = [25, 75]
        
        # [Min, Max] High frequency damping as a percentage, higher = more damping.
        self.reverbHFDampingRange = reverbHFDampingMinMax
        if reverbHFDampingMinMax is None:
            self.reverbHFDampingRange = [25, 75]
        
        # [Min, Max] How long reverb is sustained as a percentage, higher = lasts longer.
        self.reverbSustainRange = reverbSustainMinMax 
        if reverbSustainMinMax is None:
            self.reverbSustainRange = [25, 75]       

        # Audio files to use as source of noise.
        self.noiseFiles = noiseFileList
        if self.noiseFiles is None or len(self.noiseFiles) == 0:
            self.noiseAugProb = -1

        # [Min, Max] Signal to noise ratio range for adding noise to audio.
        # Lower SNR = noise is more prominent, i.e. speech is more noisy.
        self.noiseSNRRange = noiseSNRMinMax
        if noiseSNRMinMax is None:
            self.noiseSNRRange = [10.0, 30.0]
        
        self.validateCONFIG()
        
    def validateCONFIG(self):
        """
        Checks CONFIGured options and raises an error if they
        are not consistent with what is expected.
        """
        if len(self.volScaleRange) != 2:
            raise ValueError("volume scale range must be provided as [min, max]")
        if len(self.reverbRoomScaleRange) != 2:
            raise ValueError("reverb room scale range must be provided as [min, max]")
        if len(self.reverbHFDampingRange) != 2:
            raise ValueError("reverb high frequency dampling range must be provided as [min, max]")
        if len(self.reverbSustainRange) != 2:
            raise ValueError("reverb sustain range must be provided as [min, max]")
        if len(self.noiseSNRRange) != 2:
            raise ValueError("noise SNR range must be provided as [min, max]")
            
        for v in self.reverbRoomScaleRange:
            if v > 100 or v < 0:
                raise ValueError("reverb room scale must be between 0 and 100")
        for v in self.reverbHFDampingRange:
            if v > 100 or v < 0:
                raise ValueError("reverb high frequency dampling must be between 0 and 100")
        for v in self.reverbSustainRange:
            if v > 100 or v < 0:
                raise ValueError("reverb sustain range must be between 0 and 100")

    @classmethod
    def loadAudio(
        cls, audioPath: str, sampleRate: int = None, returnTensor: bool = True, resampleType: str = "fast",
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Uses torchaudio to load and resample (if necessary) audio files and returns
        audio samples as either a numpy.float32 array or a torch.Tensor.
        
        Parameters
        ----------
        audioPath: str
            Path to audio file file (wav / mp3 / flac).
        
        sampleRate: int, optional
            Sampling rate to convert audio to. If None,
            audio is not resampled.
        
        returnTensor: bool, optional
            If True, the audio samples are returned as a torch.Tensor.
            Otherwise, the samples are returned as a numpy.float32 array.
            
        resampleType: str, optional
            Either "fast" or "best" - sets the quality of resampling.
            "best" is twice as slow as "fast" but more accurate. "fast"
            is still comparable to librosa's resampled output though,
            in terms of MSE.

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            Audio waveform scaled between +/- 1.0 as either a numpy.float32 array,
            or torch.Tensor, with shape (channels, numSamples)
        """
        if audioPath.endswith("mp3"):
            x, sr = torchaudio.load(audioPath, format="mp3")
        else:
            x, sr = torchaudio.load(audioPath)

        if sampleRate is not None or sr != sampleRate:
            x = F.resample(x, sr, sampleRate, **cls.resampleFilterParams[resampleType])
            
        
        if returnTensor:
            return x
        
        return x.numpy()

    def getAudio(self, audioPath: str, returnTensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Loads audio from specified path and applies augmentations randomly
        on-the-fly. Audio samples scaled between -1.0 and +1.0 are returned
        as a numpy.float32 array or torch.Tensor with shape (numSamples,).

        Parameters
        ----------
        audioPath: str
            Path to audio file file (wav / mp3 / flac).
        
        returnTensor: bool, optional
            If True, the audio samples are returned as a torch.Tensor.
            Otherwise, the samples are returned as a numpy.float32 array.
        
        Returns
        ------- 
        Union[torch.Tensor, np.ndarray]
            Audio waveform scaled between +/- 1.0 as either a numpy.float32 array,
            or torch.Tensor, with shape (channels, numSamples)
        """
        
        wav = self.loadAudio(
            audioPath, sampleRate=self.sampleRate, returnTensor=True, resampleType="fast",
        )

        # Applying sox-based effects first.
        effects = []
        
        if random.uniform(0, 1) <= self.speedAugProb:
            effects.extend([
                ["speed", f"{random.choice(self.speedFactors)}"],
                ["rate", f"{self.sampleRate}"],
            ])

        if random.uniform(0, 1) <= self.reverbAugProb:
            effects.append([
                "reverb",
                f"{random.uniform(*self.reverbSustainRange)}",
                f"{random.uniform(*self.reverbHFDampingRange)}",
                f"{random.uniform(*self.reverbRoomScaleRange)}",
            ])
        
        # If no effects are selected, this is a no-op.
        # wav = self.applySoxEffects(wav, effects)

        if random.uniform(0, 1) <= self.noiseAugProb:
            noiseFile = random.choice(self.noiseFiles)
            noiseSNR = random.uniform(*self.noiseSNRRange)
            wav = self.addNoiseFromFile(wav, noiseFile, noiseSNR)

        if random.uniform(0, 1) <= self.volAugProb:
            volScale = random.uniform(*self.volScaleRange)
            wav = self.scaleVolume(wav, volScale)
            
        if random.uniform(0, 1) <= self.downsample_8KProb:
            wav = F.resample(wav,self.sampleRate,8000, **self.resampleFilterParams["fast"])
            wav = F.resample(wav,8000,self.sampleRate, **self.resampleFilterParams["fast"])
            
        
        if returnTensor:
            return wav
            
        
        return wav.numpy()


    def scaleVolume(self, wav: Union[np.ndarray, torch.Tensor], scale: float) -> torch.Tensor:
        """
        Scales the amplitude (with clipping) of the provided audio signal
        by the given scale factor.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).

        Returns
        -------
        torch.Tensor
            Audio samples with perturbed volume.
        """
        if scale == 1.0:
            return wav

        return torch.clamp(wav * scale, -1.0, 1.0)

    def addNoiseFromFile(
        self, wav: Union[np.ndarray, torch.Tensor], noiseFile: str, snr: float,
    ) -> torch.Tensor:
        """
        Adds noise signal from provided noise audio file at the 
        specified SNR to the speech signal.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).

        snr: float
            Signal-to-Noise ratio at which to mix in the noise signal.
        
        Returns
        -------
        torch.Tensor
            Audio samples with noise added at specified SNR.
        """
        # Loading noise signal.
        noiseSig = self.loadAudio(
            noiseFile, sampleRate=self.sampleRate, returnTensor=True, resampleType="fast",
        )

        # Computing noise power.
        noisePower = torch.mean(torch.pow(noiseSig, 2))
        
        # Computing signal power.
        signalPower = torch.mean(torch.pow(wav, 2))

        # Noise Coefficient for target SNR; amplitude coeff is sqrt of power coeff.
        noiseScale = torch.sqrt((signalPower / noisePower) / (10 ** (snr / 20.0)))
        
        # Add noise at random location in speech signal.
        nWav, nNoise = wav.shape[-1], noiseSig.shape[-1]

        if nWav < nNoise:
            a = random.randint(0, nNoise-nWav)
            b = a + nWav
            return wav + (noiseSig[..., a:b] * noiseScale)
        
        a = random.randint(0, nWav-nNoise)
        b = a + nNoise          
        wav[..., a:b] += (noiseSig * noiseScale)

        return wav
    
        
    def applySoxEffects(self, wav: Union[np.ndarray, torch.Tensor], effects: List[List[str]]) -> torch.Tensor:
        """
        Applies different audio manipulation effects to provided audio, like
        speed and volume perturbation, reverberation etc. For a full list of
        supported effects, check torchaudio.sox_effects.

        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).
        
        effects: List[List[str]]
            List of sox effects and associated arguments, example:
            '[ ["speed", "1.2"], ["vol", "0.5"] ]'

        Returns
        -------
        torch.Tensor
            Audio samples with effects applied. May not be the same
            number of samples as input sample array, depending on types
            of effects applied (e.g. speed perturbation may reduce or
            increase the number of samples).
        """
        if effects is None or len(effects) == 0:
            return wav

        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav, sample_rate=self.sampleRate, effects=effects,
        )

        return wav
    
    def perturbSpeed(self, wav: Union[np.ndarray, torch.Tensor], factor: float) -> torch.Tensor:
        """
        Perturbs the speed of the provided audio signal by the given factor.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).

        Returns
        -------
        torch.Tensor
            Audio samples with perturbed speed. Will have more or less
            samples than input depending on whether slowed down or
            sped up.
        """
        effects = [
            ["speed", f"{factor}"],
            ["rate", f"{self.sampleRate}"],
        ]
        
        return self.applySoxEffects(wav, effects)
    
    def addReverb(
        self, wav: Union[np.ndarray, torch.Tensor], roomSize: float, hfDamping: float, sustain: float,
    ) -> torch.Tensor:
        """
        Adds reverberation to the provided audio signal using given parameters.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).
        
        roomSize: float
            Room size as a percentage between 0 and 100,
            higher = more reverb

        hfDamping: float
            High Frequency damping as a percentage between 0 and 100,
            higher = more damping.

        sustain: float
            How long reverb is sustained as a percentage between 0 and 100,
            higher = lasts longer.

        Returns
        -------
        torch.Tensor
            Audio samples with reverberated audio.
        """
        effects = [["reverb", f"{roomSize}", f"{hfDamping}", f"{sustain}"]]
        return self.applySoxEffects(wav, effects)
