import pygame 
from math import pi

pygame.init()

size = [600, 600] 
screen = pygame.display.set_mode(size)
while True:
    i = 100
    screen.fill((0, 0, 0))
    vertical_line = pygame.Surface((1, 600), pygame.SRCALPHA)
    vertical_line.fill((0, 255, 0, 100)) # You can change the 100 depending on what transparency it is.
    screen.blit(vertical_line, (50, 25))

    pygame.display.flip()

pygame.quit()