import pygame
import math
import sys
import os
import numpy as np
import time
import array
import random
import warnings
import torch
from Constant import ConstPara, Action, NoteType
from Background import Background
from Sprite import NoteBarGroup
from NaiveJudge import SimpleJudgement
from ToneGenerator import EnvMap


class NaiveBandori:

    def __init__(
            self, height=144, noteSpeed=9.0,  # songNo=73, difficulty=Difficulty.expert,
            interval=3, audio_state=True, real_music=False, seed=None):
        self.const = ConstPara(height=height, noteSpeed=noteSpeed)
        self.screen = pygame.display.set_mode(self.const.size, flags=0, depth=24)  # window size
        self.bg = Background(self.const)
        # self.s_c = SongChart(songNo=songNo, difficulty=difficulty)
        self.seed = seed
        # if isinstance(self.seed, int):
        #    random.seed(self.seed)
        self._e = EnvMap(random.randint(8, 12) * 1000)
        j, self.pydub_bgm = self._e.get_chart_song()
        # e.save_and_play()
        self.ng = NoteBarGroup(j, self.const)
        self.in_time_music = real_music  # play music? default is false
        # assert not self.in_time_music
        self.playing = False
        self.delta_t = 15.5 / 1000.
        self.interval = interval
        self.audio_state = audio_state
        # for agent
        self.info = {}
        self.rec_surface = self.screen.subsurface((math.ceil(self.const.width*.43), self.const.height / 2, math.ceil(self.const.width*(1-2*.43)), self.const.height / 2))
        self.playTime = random.randint(0, 16)/1000.
        self.judge = SimpleJudgement(
            self.ng.basic_notes, self.ng.notes_hit)
        # self.pydub_bgm = pydub.AudioSegment.from_mp3(self.s_c.songPath)
        self.obs = [0 for _ in range(self.interval+(1 if self.audio_state else 0))]  # screen*3, audio segment
        # self.songNo = songNo
        # print(j)
        self.music_file = None  # for test

    def start_music(self):
        # self.pygame_audio = pygame.mixer.Sound(array=(self.pydub_bgm/10200.*126).astype(np.int8),)
        # self.pygame_audio.play(-1)
        # import struct
        # for i in self.pydub_bgm:
        # b = struct.pack('<h', (self.pydub_bgm/10200*127).astype(np.int8))
        # pygame.mixer.music.load(b)
        self._e.save_and_play()
        self.music_file = open('a.wav', 'rb')
        pygame.mixer.music.load(self.music_file)
        pygame.mixer.music.play(start=0.0)

    def update(self):
        if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy":
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        if self.playing:
                            self.playing = False
                            pygame.mixer.music.pause()
                        else:
                            self.playing = True
                            pygame.mixer.music.unpause()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        if self.in_time_music:
            self.playTime = pygame.mixer.music.get_pos() / 1000.
        else:
            self.playTime += 0.016  # 60+ frames in 1s
        self.screen.fill((0, 0, 0))
        self.bg.draw_lane(self.screen)
        self.ng.bar_update_draw(
            self.screen,
            self.playTime,
        )
        self.bg.draw_line(self.screen)
        self.ng.note_update_draw(
            self.screen,
            self.playTime,
        )
        # show song No. for debug?
        # textSurface = pygame.font.Font(None, 32)
        # text = textSurface.render(str(self.songNo), 1, (255,255,255))
        # self.screen.blit(text, (0,0))
        pygame.display.update()

    def music_slice(self, time):
        multipier = self._e.frame_rate//1000
        if time > len(self.pydub_bgm):
            p1 = self.pydub_bgm[(time-self.interval*16)*multipier:]#.get_array_of_samples()
            l = len(self.pydub_bgm[0:16*self.interval*multipier
                    ]#.get_array_of_samples()
                    )-len(p1)
            return p1+array.array('i', [0 for _ in range(l)])
        else:
            # print((time-self.interval*16)*multipier,time*multipier)
            return self.pydub_bgm[(time-self.interval*16)*multipier:time*multipier] #.get_array_of_samples()

    # adapt for openai-gym, in_time_music is False
    def step(self, act):
        if self.in_time_music:
            self.in_time_music = False
            warnings.warn('Slow training if in_time_music')
        reward = self.judge.judge(self.playTime*1000, act)
        if self.judge.no_life_now == 1 or self.playTime >= self.ng.note_end_time+1:
            # dead / finish
            # self.obs = self.reset()
            done =True
        else:
            # go on
            done = False
        for i in range(self.interval):
            self.update()
            self.obs[i] = pygame.surfarray.array3d(self.rec_surface)
        if self.audio_state:
            self.obs[-1] = self.music_slice(int(self.playTime*1000))
        # print(self.obs[-1].shape)
        return self.obs, reward, done, self.info

    def reset(self):
        self._e = EnvMap(random.randint(8, 12) * 1000)
        j, self.pydub_bgm = self._e.get_chart_song()
        self.ng = NoteBarGroup(j, self.const)
        # for agent
        self.info = {}
        self.playTime = random.randint(0, 16)/1000.
        self.judge = SimpleJudgement(
            self.ng.basic_notes, self.ng.notes_hit)
        # self.pydub_bgm = pydub.AudioSegment.from_mp3(self.s_c.songPath)
        self.obs = [0 for _ in range(self.interval+(1 if self.audio_state else 0))]  # screen*3, audio segment

        self.screen.fill((0, 0, 0))
        self.bg.draw_lane(self.screen)
        self.bg.draw_line(self.screen)
        s = pygame.surfarray.array3d(self.rec_surface)
        if not self.audio_state:
            return [s, s, s]
        else:
            return \
                [s, s, s, np.zeros_like(
                        self.pydub_bgm[0:16*self.interval*self._e.frame_rate//1000]#.get_array_of_samples()
                )]

    def show(self):
        # assert self.in_time_music
        if not self.in_time_music:
            self.in_time_music = True
            warnings.warn('Set real_music as True')
        self.start_music()
        while 0 <= self.playTime < self.ng.note_end_time+1:
            self.update()

    def set_real_music(self, b: bool):
        self.in_time_music = b

    # @staticmethod
    def show_for_act(self, actor):
        if not self.in_time_music:
            self.in_time_music = True
            warnings.warn('Set real_music as True')
        if not self.playing:
            self.start_music()
            self.playing = True
        done = False
        i = 0
        old_clock = time.time()
        acts = [None for _ in range(len(self._e.j)*2)]
        acts_index = 0
        while not done:
            self.update()
            new_clock = time.time()
            if new_clock - old_clock >= self.delta_t:
                old_clock = new_clock
                self.obs[i] = pygame.surfarray.array3d(self.rec_surface)
                i += 1
            if i == self.interval:
                # print(i)
                if self.audio_state:
                    self.obs[-1] = self.music_slice(int(self.playTime * 1000))
                # print(actor(np.array(self.obs)).squeeze(dim=0).cpu().numpy())
                if self.playTime*1000 >= 16 * self.interval:
                    with torch.no_grad():
                        act = np.argmax(actor(np.array(self.obs))[0].squeeze(dim=0).cpu().numpy())
                    _r = self.judge.judge(self.playTime*1000, act)
                    # print(act)
                    if act != Action.release.value:
                        acts[acts_index] = Action(act).name
                        acts_index += 1
                i = 0
            if self.judge.no_life_now == 1 or self.playTime >= self.ng.note_end_time + 1:
                done = True  # dead / finish
            # else:
            #     done = False  # go on
        pygame.mixer.music.stop()
        self.music_file.close()
        self.obs.clear()
        self.playing = False
        for ee in range(len(self._e.j)):
            print('Time:', self._e.j[ee]['time'],
                  'Note:', 'T' if self._e.j[ee]['type'][0]=='S' else 'F',
                  'Action:', acts[ee] if acts[ee].startswith('f') else 'tap', end=' ')
            print('√' if ('T' if self._e.j[ee]['type'][0]=='S' else 'F').lower() == (acts[ee] if acts[ee].startswith('f') else 'tap')[0] else '×')
        # print(self._e.j, acts, sep='\n')
        return self.judge.summary()
