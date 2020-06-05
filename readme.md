# Multimedia-RL in NaiveBandori

![aya](https://github.com/KurohaneNioko/Multimedia-RL/blob/master/docs/pics/interesting.png)

Both audio agents and non-audio agents are well trained.

Cli args see train_main.py

��Ҫ�޸�tianshouģ����data/buffer.py�е�_add_to_buffer������֧����״��ͬ��numpy�������ѵ������ǰtianshou�汾��0.2.2��
Plz modify data/buffer.py in module named "tianshou". In tianshou V0.2.2, function named "_add_to_buffer" should be like this to support multimedia training.
```python
def _add_to_buffer(self, name, inst):
    if inst is None:
        if getattr(self, name, None) is None:
            self.__dict__[name] = None
        return
    if self.__dict__.get(name, None) is None:
        if isinstance(inst, np.ndarray):
            self.__dict__[name] = np.zeros([self._maxsize, *inst.shape], dtype=inst.dtype)  # my modification
        elif isinstance(inst, dict):
            self.__dict__[name] = np.array(
                [{} for _ in range(self._maxsize)])
        else:  # assume "inst" is a number
            self.__dict__[name] = np.zeros([self._maxsize])
    if isinstance(inst, np.ndarray) and \
            self.__dict__[name].shape[1:] != inst.shape:
        self.__dict__[name] = np.zeros([self._maxsize, *inst.shape])
    self.__dict__[name][self._index] = inst
```