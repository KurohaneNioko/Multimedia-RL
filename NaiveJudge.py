from Constant import NoteType as nt
from Constant import Action
from enum import Enum


class NaiveGrade(Enum):
    hit, miss, redundant = [10, -10, -1]


class SimpleJudgement:

    def __init__(self, N, NH):
        self.notes, self.note_hit = N, NH
        self.frame_time = 1000. / 60.  # ms
        self.grade = [None for _ in range(len(self.notes))]
        # self.last_acts = None
        self.note_index = 0
        self.act_mark = 0
        # aim
        # self._left = -1
        # self._right = 1
        # self._miss = 0
        # summary
        self.no_life_now = 0

    def mark(self, index, grade):
        self.grade[index] = grade
        # mark note as invisible
        self.note_hit[index] = True
        return grade.value

    def hit_act(self, note_type, act):
        if note_type==nt.normal and act == Action.tap.value:
            return True
        if note_type==nt.flick and act == Action.flick.value:
            return True
        return False

    def check_life(self):
        life = 10
        for e in self.grade:
            if e: life += e.value
            if life < 0:
                self.no_life_now = 1
                break

    def judge(self, time, action):
        # action -> [(lane, act), (lane, act)]
        # earlier -> negateive, late -> positive
        # type -> nt.normal, nt.long, nt.flick, nt.slide
        # time -> ms

        # get the check range START, [note_index, note_end_index)
        for i in range(self.note_index, len(self.notes)):
            if self.grade[i] is not None:
                self.note_index += 1
                continue
            if self.grade[i] is None:   # make sure judging all notes
                break
        # get the check range END, [note_index, note_end_index)
        note_end_index, stop_point = self.note_index+1, 0
        if self.note_index < len(self.notes):
            for i in range(self.note_index, len(self.notes)):
                if not (time-self.notes[i].time) >= -8 * self.frame_time:
                    note_end_index = max(note_end_index, i)
                    break
        # duplicate for no end
        note_end_index = max(note_end_index, stop_point)
        # print(self.note_index, note_end_index, int(time), end=' || ')

        reward = 0
        # if self.last_acts is None:
        #     self.last_acts = action  # strict action ?
        if self.note_index < len(self.notes):
            self.act_mark = 0  # bit mark
            for i in range(self.note_index, note_end_index):
                # print(self.notes[i].type.value, self.notes[i].lane, self.notes[i].time, end='; ')
                t = (time - self.notes[i].time)/self.frame_time
                if self.grade[i] is None and (self.notes[i].type == nt.normal or self.notes[i].type == nt.flick):
                    if t > 3:   reward += self.mark(i, NaiveGrade.miss)
                    elif -3 <= t <= 3:
                        if self.hit_act(self.notes[i].type, action):
                            self.act_mark = 1
                            reward += self.mark(i, NaiveGrade.hit)
                        else:
                            if action != Action.release.value:  # wrong act
                                self.act_mark = 1
                                reward += self.mark(i, NaiveGrade.miss)
                if self.act_mark != 0:
                    break
        if self.act_mark == 0 and action != Action.release.value:
            reward += NaiveGrade.redundant.value
        self.check_life()
        return reward

    def summary(self):
        total_point = 0
        miss, redundant, hit = 0, 0, 0
        for e in self.grade:
            if e:
                total_point += e.value
                if e.name == NaiveGrade.miss.name:
                    miss += 1
                elif e.name == NaiveGrade.hit.name:
                    hit += 1
                elif e.name == NaiveGrade.redundant.name:
                    redundant += 1
        return '| '+str(total_point)+'pts | '+str(miss)+', '+str(redundant)+', '+str(hit)+' | length='+str(miss+redundant+hit)
