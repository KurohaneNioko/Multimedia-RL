import wave
import struct
import random
import enum
import numpy as np
import warnings

class JsonKeys(enum.Enum):
    type = 'type'
    lane = 'lane'
    time = 'time'

class SlimNoteType(enum.Enum):
    Single = 'Single'
    Flick = 'Flick'

class EnvMap:
    # generate json & audio array
    # rules: t->ms, note_time=t, audio_time=[t-duration/2, t+duration/2]
    # miss/wrong -10, tap-when-unnecessary -1, hit 10
    def __init__(self, total_time: int, volume=10000, frame_rate=48000, pitch_duration=200, end_blank=1000, seed=None):
        # total_time -> ms
        self.total_time = 1000 + (int(total_time) if total_time > 1000 else 1000*total_time)
        self.seed = seed
        # json
        self.j = []
        self.lane = 4  # one lane
        self.type_list = list(SlimNoteType)  # two kinds of notes
        # audio
        self.a = np.array([])
        self.frame_rate = frame_rate
        self.piece_duration = pitch_duration
        self.volume = volume
        self.end_blank = end_blank  # add blank audio
        self.pitch_no = [60, 62, 64, 65, 67, 69, 71]  # C4 D4 E4 F4 G4 A4 B4
        self.freq = list(map(lambda x: 440*(2**((x-69)/12.)), self.pitch_no))
        # run
        self._gen_()

    def _json_(self):
        pitch_interval = int(self.piece_duration*1.05)
        t = 2000
        t += random.randint(0, 1000)  # start point in [3000, 4000]
        while t < self.total_time:
            self.j.append({
                JsonKeys.type.value: self.type_list[random.randint(0,1)].value,
                JsonKeys.lane.value: self.lane,
                JsonKeys.time.value: t/1000.
            })
            if t + pitch_interval < self.total_time:
                t += random.randint(pitch_interval, min(pitch_interval*2, self.total_time-t, ))
            else:
                break

    def _audio_(self):
        # Single -> do re mi fa, Flick -> fa so la si
        audio_with_time_stamp = []
        for note in self.j:
            if note[JsonKeys.type.value] == SlimNoteType.Flick.value:
                f = self.freq[random.randint(len(self.pitch_no)//2, len(self.pitch_no)-1)]
            elif note[JsonKeys.type.value] == SlimNoteType.Single.value:
                f = self.freq[random.randint(0, len(self.pitch_no)//2)]
            else:
                f = self.freq[len(self.pitch_no)//2]
                warnings.warn('Invalid note type')
            # print(f)
            audio_clip = (self.volume * np.sin(
                2*np.pi*f*np.linspace(0, self.piece_duration/1000, num=(self.piece_duration*self.frame_rate)//1000)))
            audio_with_time_stamp.append([note[JsonKeys.time.value]-(self.piece_duration>>1)/1000., audio_clip])
        last_t = 0
        for e in audio_with_time_stamp:
            t, a_clip = e  # audio starting timr
            t = int(t*1000)
            # print(t)
            self.a = np.concatenate(
                (self.a, np.zeros(int((t-last_t)*self.frame_rate/1000)), a_clip),
                axis=-1
            )
            # print(len(np.zeros(int((t-last_t)*self.frame_rate/1000)))/self.frame_rate*1000)
            last_t = t+self.piece_duration
        self.a = np.concatenate((self.a, np.zeros(int(self.frame_rate*self.end_blank/1000))), axis=-1)
        # add white noise
        white_noise = np.random.random(len(self.a))
        # self.a = np.where(self.a != 0.0, self.a + 0.02 * white_noise, 0.0).astype(np.float32)
        self.a += 0.02 * self.volume * white_noise

    def _gen_(self):
        if self.seed is not None:
            random.seed(self.seed)
        self._json_()
        self._audio_()
        import gc
        gc.collect()

    def get_chart_song(self):
        return self.j, self.a

    def save_and_play(self):
        if len(self.j)>0 and (len(self.a))>0:
            # print(self.j)
            sample_width = 2  # 2 bytes(half-int) per sample
            wf = wave.open("a.wav", 'wb')
            wf.setnchannels(1)
            wf.setframerate(self.frame_rate)
            wf.setsampwidth(sample_width)
            for i in self.a:
                data = struct.pack('<h', int(i))
                wf.writeframesraw(data)
            wf.close()
        else:
            warnings.warn('Not run yet')

# e = EnvMap(8*1000)
# e.save_and_play()
#
# framerate = 48000
# duration = 0.512
# frequency = 349.22
# volume = 10000
# x = np.linspace(0, duration, num=int(duration*framerate), dtype=np.float64)
# y = np.sin(2 * np.pi * frequency * x) * volume
# sine_wave = y.astype(np.int16)
# sample_width = 2  # 2**4 bit
# wf = wave.open("sine.wav", 'wb')
# wf.setnchannels(1)
# wf.setframerate(framerate)
# wf.setsampwidth(sample_width)
# for i in sine_wave:
#     data = struct.pack('<h', i)
#     wf.writeframesraw(data)
# wf.close()

# import pydub
# m = pydub.AudioSegment.from_wav('sine.wav',
#         parameters={'sample_width': sample_width, 'frame_rate': framerate, 'channels': 1})
# t = np.array(m[:].get_array_of_samples())
# print(m.get_array_of_samples())
