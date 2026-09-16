import pygame
import time

pygame.init()
pygame.joystick.init()

print("Controllers:", pygame.joystick.get_count())

pad = pygame.joystick.Joystick(0)
pad.init()

print("Name:", pad.get_name())
print("Axes:", pad.get_numaxes())
print("Buttons:", pad.get_numbuttons())

while True:
    pygame.event.pump()

    axes = [
        round(pad.get_axis(i), 3)
        for i in range(pad.get_numaxes())
    ]

    buttons = [
        pad.get_button(i)
        for i in range(pad.get_numbuttons())
    ]

    print("axes:", axes, "buttons:", buttons)

    time.sleep(0.05)