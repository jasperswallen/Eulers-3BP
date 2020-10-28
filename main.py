import matplotlib.pyplot as plt
import numpy as np

mass_1_x = float(input("Enter M1's x position: "))
mass_1_y = float(input("Enter M1's y position: "))
mass_1 = float(input("Enter M1's mass: "))
mass_2_x = float(input("Enter M2's x position: "))
mass_2_y = float(input("Enter M2's y position: "))
mass_2 = float(input("Enter M2's mass: "))

alpha = mass_2 / mass_1
print(
    f"Representing the masses as factors of M1 gives us M1 = 1 * M and M2 = {alpha} * M")

moving_mass_x = float(input("Enter the moving mass's initial x position: "))
moving_mass_y = float(input("Enter the moving mass's initial y position: "))
moving_mass_x_prime = float(
    input("Enter the moving mass's initial x velocity: "))
moving_mass_y_prime = float(
    input("Enter the moving mass's initial y velocity: "))


def reorientAxes(mass_1_x, mass_1_y, mass_2_x, mass_2_y, moving_mass_x, moving_mass_y):
    print(
        f"original: m1: ({mass_1_x}, {mass_1_y}), m2: ({mass_2_x}, {mass_2_y}), m: ({moving_mass_x}, {moving_mass_y})")

    # Reposition the masses such that M1 is at the origin and M2 lies along the y-axis
    # First, move all points
    # Then rotate

    mass_2_x -= mass_1_x
    mass_2_y -= mass_1_y
    moving_mass_x -= mass_1_x
    moving_mass_y -= mass_1_y

    m_theta = np.arctan2(moving_mass_y, moving_mass_x)
    m2_theta = np.arctan2(mass_2_y, mass_2_x)

    mass_2_x = np.sqrt(mass_2_x ** 2 + mass_2_y ** 2)
    mass_2_y = 0
    mass_1_x = 0
    mass_1_y = 0

    # Subtract m2_theta from current m_theta

    m_theta -= m2_theta
    m_r = np.sqrt(moving_mass_x ** 2 + moving_mass_y ** 2)

    # Convert back to cartesian coordinates

    moving_mass_x = m_r * np.cos(m_theta)
    moving_mass_y = m_r * np.sin(m_theta)

    # m1 and m2 should now lay along the x-axis and m should be rotated relative to them
    print(
        f"final: m1: ({mass_1_x}, {mass_1_y}), m2: ({mass_2_x}, {mass_2_y}), m: ({moving_mass_x}, {moving_mass_y})")

    return (mass_1_x, mass_1_y, mass_2_x, mass_2_y, moving_mass_x, moving_mass_y)


its = int(input("Enter the number of iterations to perform: "))
delta = float(input("Enter the δt factor (the length of each input): "))


def calculateXAccel(x_val, y_val):
    return float(- 4 * (np.pi ** 2) * (x_val / np.power((x_val ** 2) + (y_val ** 2), 3 / 2) + alpha * (x_val - mass_2_x) / np.power(((x_val - mass_2_x) ** 2) + (y_val ** 2), 3 / 2)))


def calculateYAccel(x_val, y_val):
    return float(- 4 * (np.pi ** 2) * y_val * (1 / np.power((x_val ** 2) + (y_val ** 2), 3 / 2) + 1 / np.power(((x_val - mass_2_x) ** 2) + (y_val ** 2), 3 / 2)))


def eulerMethod(mass_1_x, mass_1_y, mass_2_x, mass_2_y, alpha, moving_mass_x, moving_mass_y, moving_mass_x_prime, moving_mass_y_prime, its, delta):
    # declare arrays that will be appended to in loop
    x = [moving_mass_x]
    y = [moving_mass_y]
    x_prime = [moving_mass_x_prime]
    y_prime = [moving_mass_y_prime]

    # We have the following equations of motion
    # x'_(n+1)=x'_n-deltat4pi^2 (x_(n+1)/(r_1^3)_(n+1) +alpha(x_(n+1)-d)/(r_2^3)_(n+1))
    # y'_(n+1)=y'_n-deltat4pi^2 y_(n+1) (1/(r_1^3)_(n+1) +alpha/(r_2^3)_(n+1))

    for n in range(its):
        # calculate next x, y, x', and y'
        x_n_plus_one = x[n] + delta * x_prime[n]
        x.append(x_n_plus_one)
        y_n_plus_one = y[n] + delta * y_prime[n]
        y.append(y_n_plus_one)

        x_prime_n_plus_one = x_prime[n] + delta * \
            calculateXAccel(x[n + 1], y[n + 1])
        x_prime.append(x_prime_n_plus_one)

        y_prime_n_plus_one = y_prime[n] + delta * \
            calculateYAccel(x[n + 1], y[n + 1])
        y_prime.append(y_prime_n_plus_one)

    plt.plot(x, y)

    masses = [[mass_1_x, mass_2_x, moving_mass_x],
              [mass_1_y, mass_2_y, moving_mass_y]]
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

    plt.scatter(masses[0], masses[1], c=colors / 255)

    labels = ["m1", "m2", "m"]
    for i, txt in enumerate(labels):
        plt.annotate(txt, (masses[0][i], masses[1][i]))


def rungeKutta(mass_1_x, mass_1_y, mass_2_x, mass_2_y, alpha, moving_mass_x, moving_mass_y, moving_mass_x_prime, moving_mass_y_prime, its, delta):
    # declare arrays that will be appended to in loop
    x = [moving_mass_x]
    y = [moving_mass_y]
    x_prime = [moving_mass_x_prime]
    y_prime = [moving_mass_y_prime]

    # we have y_(n+1)=y_n+deltat/6(k_1+2k_2+2k_3+k_4)

    for n in range(its):
        # calculate next x, y, x', and y'
        x_1 = delta * x_prime[n]
        y_1 = delta * y_prime[n]
        x_prime_1 = delta * calculateXAccel(x[n], y[n])
        y_prime_1 = delta * calculateYAccel(x[n], y[n])

        x_2 = delta * (x_prime[n] + x_prime_1 / 2)
        y_2 = delta * (y_prime[n] + y_prime_1 / 2)
        x_prime_2 = delta * calculateXAccel(x[n] + x_1 / 2, y[n] + y_1 / 2)
        y_prime_2 = delta * calculateYAccel(x[n] + x_1 / 2, y[n] + y_1 / 2)

        x_3 = delta * (x_prime[n] + x_prime_2 / 2)
        y_3 = delta * (y_prime[n] + y_prime_2 / 2)
        x_prime_3 = delta * calculateXAccel(x[n] + x_2 / 2, y[n] + y_2 / 2)
        y_prime_3 = delta * calculateYAccel(x[n] + x_2 / 2, y[n] + y_2 / 2)

        x_4 = delta * (x_prime[n] + x_prime_3)
        y_4 = delta * (y_prime[n] + y_prime_3)
        x_prime_4 = delta * calculateXAccel(x[n] + x_3, y[n] + y_3)
        y_prime_4 = delta * calculateYAccel(x[n] + x_3, y[n] + y_3)

        x_n_plus_one = x[n] + 1 / 6 * (x_1 + 2 * x_2 + 2 * x_3 + x_4)
        y_n_plus_one = y[n] + 1 / 6 * (y_1 + 2 * y_2 + 2 * y_3 + y_4)
        x_prime_n_plus_one = x_prime[n] + 1 / 6 * \
            (x_prime_1 + 2 * x_prime_2 + 2 * x_prime_3 + x_prime_4)
        y_prime_n_plus_one = y_prime[n] + 1 / 6 * \
            (y_prime_1 + 2 * y_prime_2 + 2 * y_prime_3 + y_prime_4)

        x.append(x_n_plus_one)
        y.append(y_n_plus_one)
        x_prime.append(x_prime_n_plus_one)
        y_prime.append(y_prime_n_plus_one)

    # print(x, y, x_prime, y_prime)

    plt.plot(x, y)

    masses = [[mass_1_x, mass_2_x, moving_mass_x],
              [mass_1_y, mass_2_y, moving_mass_y]]
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

    plt.scatter(masses[0], masses[1], c=colors / 255)

    labels = ["m1", "m2", "m"]
    for i, txt in enumerate(labels):
        plt.annotate(txt, (masses[0][i], masses[1][i]))


def yoshida4thOrder(mass_1_x, mass_1_y, mass_2_x, mass_2_y, alpha, moving_mass_x,
                    moving_mass_y, moving_mass_x_prime, moving_mass_y_prime, its, delta):
    # declare arrays that will be appended to in loop
    x = [moving_mass_x]
    y = [moving_mass_y]
    x_prime = [moving_mass_x_prime]
    y_prime = [moving_mass_y_prime]

    # Yoshida's 4th Order equation

    # coefficients that stay the same each iteration
    beta = np.power(2, 1 / 3)
    c_1 = 1 / (2 * (2 - beta))
    c_2 = (1 - beta) / (2 * (2 - beta))
    c_3 = c_2
    c_4 = c_1

    d_1 = 1 / (2 - beta)
    d_2 = -beta / (2 - beta)
    d_3 = d_1
    d_4 = 0

    for n in range(its):
        x_1 = x[n] + c_1 * x_prime[n] * delta
        y_1 = y[n] + c_1 * y_prime[n] * delta

        x_prime_1 = x_prime[n] + d_1 * calculateXAccel(x_1, y_1) * delta
        y_prime_1 = y_prime[n] + d_1 * calculateYAccel(x_1, y_1) * delta

        x_2 = x_1 + c_2 * x_prime_1 * delta
        y_2 = y_1 + c_2 * y_prime_1 * delta

        x_prime_2 = x_prime_1 + d_2 * calculateXAccel(x_2, y_2) * delta
        y_prime_2 = y_prime_1 + d_2 * calculateYAccel(x_2, y_2) * delta

        x_3 = x_2 + c_3 * x_prime_2 * delta
        y_3 = y_2 + c_3 * y_prime_2 * delta

        x_prime_3 = x_prime_2 + d_3 * calculateXAccel(x_3, y_3) * delta
        y_prime_3 = y_prime_2 + d_3 * calculateYAccel(x_3, y_3) * delta

        x_4 = x_3 + c_4 * x_prime_3 * delta
        y_4 = y_3 + c_4 * y_prime_3 * delta

        x_prime_4 = x_prime_3 + d_4 * calculateXAccel(x_4, y_4) * delta
        y_prime_4 = y_prime_3 + d_4 * calculateYAccel(x_4, y_4) * delta

        x.append(x_4)
        y.append(y_4)
        x_prime.append(x_prime_4)
        y_prime.append(y_prime_4)

    plt.plot(x, y)

    masses = [[mass_1_x, mass_2_x, moving_mass_x],
              [mass_1_y, mass_2_y, moving_mass_y]]
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

    plt.scatter(masses[0], masses[1], c=colors / 255)

    labels = ["m1", "m2", "m"]
    for i, txt in enumerate(labels):
        plt.annotate(txt, (masses[0][i], masses[1][i]))


(mass_1_x, mass_1_y, mass_2_x, mass_2_y, moving_mass_x, moving_mass_y) = reorientAxes(mass_1_x, mass_1_y, mass_2_x,
                                                                                      mass_2_y, moving_mass_x, moving_mass_y)

eulerMethod(mass_1_x, mass_1_y, mass_2_x, mass_2_y, alpha, moving_mass_x,
            moving_mass_y, moving_mass_x_prime, moving_mass_y_prime, its, delta)

yoshida4thOrder(mass_1_x, mass_1_y, mass_2_x, mass_2_y, alpha, moving_mass_x,
                moving_mass_y, moving_mass_x_prime, moving_mass_y_prime, its, delta)

rungeKutta(mass_1_x, mass_1_y, mass_2_x, mass_2_y, alpha, moving_mass_x,
           moving_mass_y, moving_mass_x_prime, moving_mass_y_prime, its, delta)

plt.show()
