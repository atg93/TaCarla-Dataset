import numpy as np
import matplotlib.pyplot as plt

def drawing_graph(x, y, x2, y2, name):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot data
    ax.plot(x, y, label='p(x) graph')

    # Set a title and labels
    ax.set_title(name)
    ax.set_xlabel('x values')
    ax.set_ylabel('y values')

    ax.plot(x2, y2, label='g(x) graph')

    #x, p = sample_function()

    # Adding a legend to distinguish the two lines
    ax.legend()

    plt.grid(True)

    #plt.show()
    plt.savefig('my_graph.png')

# Define the target distribution (that we want to estimate)
def target_distribution(x):
    return np.exp(-x ** 2 / 2)  # Example: standard normal distribution


# Define the proposal distribution (from which we can easily sample)
def proposal_distribution(x):
    return np.exp(-(x-0.2) ** 2 / 4)  # Example: normal distribution with variance 2


# Function to generate samples from the proposal distribution
def sample_from_proposal(n):
    return np.random.normal(0, 0.5, n)

def sample_function():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Parameters for the distribution
    mean = 0
    std_dev = 0.5

    # Generate x values
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)

    # Calculate the y values of the probability density function for the normal distribution
    y = norm.pdf(x, mean, std_dev)

    # Create the plot
    plt.plot(x, y, '-')

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('f(x)')

    # Display the plot
    #plt.show()
    plt.savefig('f_x.png')


# Importance sampling function
def importance_sampling(target, proposal, sample_from_proposal, num_samples):
    samples = sample_from_proposal(num_samples)
    weights = target(samples) / proposal(samples)
    estimate = np.mean(weights)
    variance = np.var(weights) / num_samples  # Variance of importance sampling estimator
    return estimate, variance

def wo_importance_sampling(target, sample_from_proposal, num_samples):
    samples = sample_from_proposal(num_samples)
    weights = target(samples)
    estimate = np.mean(weights)
    variance = np.var(weights) / num_samples  # Variance of importance sampling estimator
    return estimate, variance

#sample_function()

x_input = np.arange(-10,10,0.1)
y_output = target_distribution(x_input)
y_output_2 = proposal_distribution(x_input)
drawing_graph(x_input, y_output, x_input, y_output_2,  name='')
# Number of samples
num_samples = 10000
# Perform importance sampling
is_estimate, is_variance = importance_sampling(target_distribution, proposal_distribution, sample_from_proposal, num_samples)
print(f'Estimate of the expected value: {is_estimate}')
print(f'Variance of the weights: {is_variance}')
wo_estimate, wo_variance = wo_importance_sampling(target_distribution, sample_from_proposal, num_samples)
print(f'Estimate of the expected value wo_importance_sampling: {wo_estimate}')
print(f'Variance of the weights wo_importance_sampling: {wo_variance}')

print(f'ratio of variances: {wo_variance/is_variance}')

#sample_function()