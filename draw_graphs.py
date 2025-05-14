import matplotlib.pyplot as plt
import math
import numpy as np


x_axis = np.array(list(range(100)))[1:]
#y_axis_1 = x_axis np.ones(len(x_axis))# #np.log(x_axis)
y_axis = np.sqrt(np.log(x_axis)/x_axis) # #np.log(x_axis)


# Sample data for x and y
x = list(x_axis)
y = list(y_axis)

# Create the plot
plt.plot(x, y, marker='o')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('UCB')

# Display the graph #draw_graphs_image
#plt.show()
figure_path = "draw_graphs_image/UCB.png"
plt.savefig(figure_path)