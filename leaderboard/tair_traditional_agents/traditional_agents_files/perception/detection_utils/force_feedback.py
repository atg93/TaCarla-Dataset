import evdev
from evdev import ecodes, InputDevice, ff
import math
class ForceFeedback():
    def __init__(self, deviceNumber):
        self.device = evdev.list_devices()[deviceNumber]
        self.evtdev = InputDevice(self.device)
        self.left_effect = None
        self.right_effect = None
    def create_right_effect(self, level):
        self.right_effect = ff.Effect(
            ecodes.FF_CONSTANT, -1, 0xc000,
            ff.Trigger(0,0),
            ff.Replay(0,0),
            ff.EffectType(ff_constant_effect=ff.Constant(level=level))
        )
        self.right_effect.id = self.evtdev.upload_effect(self.right_effect)


    def create_left_effect(self, level):
        self.left_effect = ff.Effect(
            ecodes.FF_CONSTANT, -1, 0x4000,
            ff.Trigger(0, 0),
            ff.Replay(0, 0),
            ff.EffectType(ff_constant_effect=ff.Constant(level=level))
        )
        self.left_effect.id = self.evtdev.upload_effect(self.left_effect)


    def autocenter_wheel(self, val=6000):
        self.evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

    def left_turn_wheel(self, val):
        self.erase_effect(True)
        turn_value = min(2000,int((val * 0x7fff) / 2)) #3e bolunce oldu
        self.create_left_effect(turn_value)
        self.evtdev.write(ecodes.EV_FF, self.left_effect.id, 1)

    def right_turn_wheel(self, val):
        self.erase_effect(False)
        turn_value = min(2000,int((val * 0x7fff) / 2))
        self.create_right_effect(turn_value)
        self.evtdev.write(ecodes.EV_FF, self.right_effect.id, 1)

    def done_right(self):
        if self.right_effect is not None:
            self.evtdev.write(ecodes.EV_FF, self.right_effect.id, 0)

    def done_left(self):
        if self.left_effect is not None:
            self.evtdev.write(ecodes.EV_FF, self.left_effect.id, 0)

    def erase_effect(self, left):
        if left and self.left_effect is not None:
            self.evtdev.erase_effect(self.left_effect.id)
            self.left_effect = None
        elif left is False and self.right_effect is not None:
            self.evtdev.erase_effect(self.right_effect.id)
            self.right_effect = None

    def turn_wheel_angle(self, steer):
        val = min(int((steer * 32768) + 32768), 65535)
        self.evtdev.write(ecodes.EV_ABS, ecodes.ABS_X, val)

    def erase_turn_angle(self):
        self.evtdev.erase_effect(ecodes.ABS_X)

    def erase_autocenter(self):
        self.evtdev.erase_effect(ecodes.FF_AUTOCENTER)