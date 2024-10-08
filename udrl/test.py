# import gymnasium as gym
# import pygame
# import numpy as np


# def normalize_value(value, is_bounded, low=None, high=None):
#     if is_bounded:
#         return (value - low) / (high - low)
#     else:
#         return 0.5 * (np.tanh(value / 2) + 1)


# def draw_bar(screen, start, value, max_length, color, height=20):
#     bar_length = value * max_length
#     pygame.draw.rect(screen, color, (*start, bar_length, height))
#     pygame.draw.rect(
#         screen, (0, 0, 0), (*start, max_length, height), 2
#     )  # Border
#     mid_x = start[0] + max_length / 2
#     pygame.draw.line(
#         screen, (0, 0, 0), (mid_x, start[1]), (mid_x, start[1] + height), 2
#     )


# def visualize_environment(screen, state, env):
#     screen_width, screen_height = screen.get_size()
#     screen.fill((255, 255, 255))

#     # Visualize environment-specific elements
#     if env.spec.id.startswith("CartPole"):
#         cart_x = int(state[0] * 50 + screen_width // 2)
#         cart_y = screen_height - 100
#         pole_angle = state[2]
#         pygame.draw.rect(screen, (0, 0, 0), (cart_x - 30, cart_y - 15, 60, 30))
#         pygame.draw.line(
#             screen,
#             (0, 0, 0),
#             (cart_x, cart_y),
#             (
#                 cart_x + int(np.sin(pole_angle) * 100),
#                 cart_y - int(np.cos(pole_angle) * 100),
#             ),
#             6,
#         )
#     elif env.spec.id.startswith("Acrobot"):
#         center_x, center_y = screen_width // 2, screen_height // 2
#         l1, l2 = 100, 100  # Length of links
#         s0, s1 = state[0], state[1]  # sin(theta1), sin(theta2)
#         c0, c1 = state[2], state[3]  # cos(theta1), cos(theta2)
#         x0, y0 = center_x, center_y
#         x1 = x0 + l1 * s0
#         y1 = y0 + l1 * c0
#         x2 = x1 + l2 * s1
#         y2 = y1 + l2 * c1
#         pygame.draw.line(screen, (0, 0, 0), (x0, y0), (x1, y1), 6)
#         pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 6)
#         pygame.draw.circle(screen, (0, 0, 255), (int(x0), int(y0)), 10)
#         pygame.draw.circle(screen, (0, 255, 0), (int(x1), int(y1)), 10)
#         pygame.draw.circle(screen, (255, 0, 0), (int(x2), int(y2)), 10)
#     # Add more environment-specific visualizations here as needed

#     # Draw bars for each state dimension
#     num_dims = env.observation_space.shape[0]
#     bar_colors = [
#         (255, 0, 0),
#         (0, 255, 0),
#         (0, 0, 255),
#         (255, 255, 0),
#         (255, 0, 255),
#         (0, 255, 255),
#     ]
#     bar_starts = [(50, 50 + i * 70) for i in range(num_dims)]
#     max_length = 300

#     for i, (start, color) in enumerate(zip(bar_starts, bar_colors)):
#         is_bounded = not (
#             env.observation_space.high[i] > 100
#         ) and not np.isinf(env.observation_space.low[i] < -100)
#         normalized_value = normalize_value(
#             state[i],
#             is_bounded,
#             env.observation_space.low[i],
#             env.observation_space.high[i],
#         )
#         draw_bar(screen, start, normalized_value, max_length, color)

#         # Draw labels
#         font = pygame.font.Font(None, 30)
#         text = font.render(f"Dim {i}: {state[i]:.2f}", True, (0, 0, 0))
#         screen.blit(text, (start[0], start[1] - 30))

#         # Add description of bar representation
#         if is_bounded:
#             desc = f"(Range: {env.observation_space.low[i]:.2f} to {env.observation_space.high[i]:.2f})"
#         else:
#             desc = "(Unbounded: Center is 0, edges are ±∞)"
#         desc_text = pygame.font.Font(None, 24).render(
#             desc, True, (100, 100, 100)
#         )
#         screen.blit(desc_text, (start[0], start[1] + 25))

#     pygame.display.flip()


# def run_visualization(env_name):
#     pygame.init()
#     screen = pygame.display.set_mode((800, 600))
#     pygame.display.set_caption(f"{env_name} Visualization")

#     env = gym.make(env_name)
#     state, _ = env.reset()

#     clock = pygame.time.Clock()

#     running = True
#     while running:
#         visualize_environment(screen, state, env)
#         action = env.action_space.sample()
#         state, reward, done, truncated, info = env.step(action)

#         if done or truncated:
#             state, _ = env.reset()

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         clock.tick(60)  # Limit to 60 FPS

#     env.close()
#     pygame.quit()


# # Example usage
# # run_visualization("CartPole-v1")
# # Uncomment the line below to run Acrobot visualization
# run_visualization("Acrobot-v1")
