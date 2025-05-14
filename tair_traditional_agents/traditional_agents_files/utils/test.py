import evdev
from evdev import ecodes, InputDevice, ff
import time
def create_right_effect(level = 0x7fff):
	right_effect = ff.Effect(
		ecodes.FF_CONSTANT, -1, 0xc000,
		ff.Trigger(0,0),
		ff.Replay(0,0),
		ff.EffectType(ff_constant_effect=ff.Constant(level=level))
	)
	right_effect.id = evtdev.upload_effect(right_effect)
	return right_effect
	
	

device = evdev.list_devices()[0]
evtdev = InputDevice(device)

val = 24000
#evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

right_val = int(2000)
print("right value = ", right_val)
right_effect = create_right_effect(right_val)

#evtdev.upload_effect(right_effect)
evtdev.write(ecodes.EV_FF, right_effect.id, 1)
time.sleep(0.5)
evtdev.erase_effect(right_effect.id)



