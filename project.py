import numpy as np
import simpy
import matplotlib.pyplot as plt

class Queue1():
    """Represents the first queue in a tandem queueing system.

    This class simulates a single-server queue with customers arriving according to a Poisson process
    and being served with exponentially distributed service times. Customers leaving this queue
    are sent to the next queue in the system.

    Attributes:
        env: The SimPy environment for the simulation.
        servers: A SimPy Resource representing the server in the queue.
        lambda_: The arrival rate of customers (λ).
        mu: The service rate of the server (μ).
        next_queue: The next queue in the tandem system.
        customer_count: The total number of customers that have arrived at the queue.
        customer_served: The total number of customers that have been served by the queue.
        queue_length_over_time: A list of tuples representing the queue length at each time step.
        queue_length_at_events: A list of tuples representing the queue length at each event (arrival, departure, service start).
        log: A boolean flag indicating whether to print log messages during the simulation.
    """
    def __init__(self, env, capacity, lambda_, mu, next_queue, log):
        """Initializes a Queue1 object.

        Args:
            env: The SimPy environment for the simulation.
            capacity: The capacity of the server (number of customers it can serve simultaneously).
            lambda_: The arrival rate of customers (λ).
            mu: The service rate of the server (μ).
            next_queue: The next queue in the tandem system.
            log: A boolean flag indicating whether to print log messages during the simulation.
        """
        self.env = env
        self.servers = simpy.Resource(env, capacity)
        self.lambda_ = lambda_
        self.mu = mu
        self.next_queue = next_queue
        self.customer_count = 0
        self.customer_served = 0
        self.queue_length_over_time = []
        self.queue_length_at_events = []  # Store queue lengths at events
        self.log = log

    def serve(self, id):
        """Simulates the service of a customer at the queue.

        This method generates a random service time based on the exponential distribution with rate μ.
        After the service is complete, the customer is sent to the next queue.

        Args:
            id: The ID of the customer being served.
        """
        random_service_time = np.random.exponential(1.0/self.mu)
        yield self.env.timeout(random_service_time)
        if self.log:
            print(f"Queue 1: Customer {id} leave service at {self.env.now}")
        self.customer_served += 1
        self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))
        # Schedule the customer's arrival at the next queue as a process
        yield self.env.process(self.next_queue.arrive(id))  

    def arrive(self):
        """Simulates the arrival of customers at the queue.

        This method generates random inter-arrival times based on the exponential distribution with rate λ.
        For each arrival, a new customer process is created.
        """
        while True:
            random_arrival_time = np.random.exponential(1.0/self.lambda_)
            yield self.env.timeout(random_arrival_time)
            self.customer_count += 1
            self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))
            self.env.process(self.customer(self.customer_count))

    def customer(self, id, count_in=False):
        """Simulates the arrival and service of a customer at the queue.

        This method represents the complete process of a customer arriving, waiting for service,
        being served, and then moving to the next queue.

        Args:
            id: The ID of the customer.
            count_in: A boolean flag indicating whether to increment the customer count.
        """
        if (count_in):
            self.customer_count += 1
            id = self.customer_count
        arrival_time = self.env.now
        if self.log:
            print(f"Queue 1: Customer {id} arrived at {arrival_time}")
        self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))
        with self.servers.request() as request:
            yield request
            if self.log:
                print(f"Queue 1: Customer {id} enter service at {self.env.now}")
            self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))
            yield self.env.process(self.serve(id))

    def monitor(self):
        """Monitors the queue length over time.

        This method records the queue length at each time step (every 1 unit of time).
        """
        while True:
            self.queue_length_over_time.append((self.env.now, self.customer_count - self.customer_served))
            yield self.env.timeout(1)

class Queue2():
    """Represents the second queue in a tandem queueing system.

    This class simulates a single-server queue with customers arriving from the previous queue
    and being served with exponentially distributed service times. Customers leaving this queue
    exit the system.

    Attributes:
        env: The SimPy environment for the simulation.
        servers: A SimPy Resource representing the server in the queue.
        mu: The service rate of the server (μ).
        customer_count: The total number of customers that have arrived at the queue.
        customer_served: The total number of customers that have been served by the queue.
        queue_length_over_time: A list of tuples representing the queue length at each time step.
        queue_length_at_events: A list of tuples representing the queue length at each event (arrival, departure, service start).
        log: A boolean flag indicating whether to print log messages during the simulation.
    """
    def __init__(self, env, capacity, mu, log):
        """Initializes a Queue2 object.

        Args:
            env: The SimPy environment for the simulation.
            capacity: The capacity of the server (number of customers it can serve simultaneously).
            mu: The service rate of the server (μ).
            log: A boolean flag indicating whether to print log messages during the simulation.
        """
        self.env = env
        self.servers = simpy.Resource(env, capacity)
        self.mu = mu
        self.customer_count = 0
        self.customer_served = 0
        self.queue_length_over_time = []
        self.queue_length_at_events = []  # Store queue lengths at events
        self.log = log
    def service(self, id):
        """Simulates the service of a customer at the queue.

        This method generates a random service time based on the exponential distribution with rate μ.
        After the service is complete, the customer leaves the system.

        Args:
            id: The ID of the customer being served.
        """
        random_service_time = np.random.exponential(1.0/self.mu)
        yield self.env.timeout(random_service_time)
        if self.log:
            print(f"Queue 2: Customer {id} leave service at {self.env.now}")
        self.customer_served += 1
        self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))

    def arrive(self, id=None):
        """Simulates the arrival of a customer at the queue.

        This method increments the customer count and creates a new customer process.

        Args:
            id: The ID of the customer arriving (optional, used to track customers from the previous queue).
        """
        self.customer_count += 1
        self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))
        # Schedule the customer's arrival at the next queue as a process
        yield self.env.process(self.customer(self.customer_count, id))

    def customer(self, id, previous_id=None):
        """Simulates the arrival and service of a customer at the queue.

        This method represents the complete process of a customer arriving, waiting for service,
        being served, and then leaving the system.

        Args:
            id: The ID of the customer.
            previous_id: The ID of the customer from the previous queue (optional).
        """
        arrival_time = self.env.now
        if self.log:
            if (previous_id is not None):
                print(f"Queue 2: Customer {previous_id} arrived from Queue 1 at {arrival_time}")
            else:
                print(f"Queue 2: Customer {id} arrived at {arrival_time}")
        self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))
        with self.servers.request() as request:
            yield request
            if self.log:
                print(f"Queue 2: Customer {id} enter service at {self.env.now}")
            self.queue_length_at_events.append((self.env.now, self.customer_count - self.customer_served))
            yield self.env.process(self.service(id))

    def monitor(self):
        """Monitors the queue length over time.

        This method records the queue length at each time step (every 1 unit of time).
        """
        while True:
            self.queue_length_over_time.append((self.env.now, self.customer_count - self.customer_served))
            yield self.env.timeout(1)

def setup(env, lambda_, mu1, mu2, capacity1, capacity2, q, log):
    """Sets up the tandem queueing system for simulation.

    This function creates the two queues and initializes them with the specified parameters.

    Args:
        env: The SimPy environment for the simulation.
        lambda_: The arrival rate of customers (λ).
        mu1: The service rate of the server in the first queue (μ1).
        mu2: The service rate of the server in the second queue (μ2).
        capacity1: The capacity of the server in the first queue.
        capacity2: The capacity of the server in the second queue.
        q: The initial number of customers in the first queue.
        log: A boolean flag indicating whether to print log messages during the simulation.

    Returns:
        A tuple containing the two Queue objects (queue1, queue2).
    """
    queue2 = Queue2(env, capacity2, mu2, log)
    queue1 = Queue1(env, capacity1, lambda_, mu1, queue2, log)

    for i in range(q):
        env.process(queue1.customer(i, count_in=True))

    env.process(queue1.arrive())
    env.process(queue1.monitor())

    return (queue1, queue2)

def calculate_time_average(points):
    """Calculates the time average of a list of points using the trapezoidal rule.

    This function takes a list of points, where each point is a tuple (time, value), and calculates
    the time average of the values using the trapezoidal rule.

    Args:
        points: A list of tuples representing the points (time, value).

    Returns:
        The time average of the values.
    """
    sum = 0.0
    # queue_length_at_events
    for i in range(len(points) - 1):
        time1 = points[i][0]
        time2 = points[i + 1][0]
        length1 = points[i][1]
        length2 = points[i + 1][1]
        sum = sum + (0.5) * (time2 - time1) * (length1 + length2) 
    return sum

def run(lambda_, mu1, mu2, capacity1, capacity2, q, T, N=10,log=True):
    """Runs the simulation of the tandem queueing system.

    This function performs multiple simulation runs and calculates the average time average number of customers
    in the system. It also plots the queue lengths over time for one simulation run.

    Args:
        lambda_: The arrival rate of customers (λ).
        mu1: The service rate of the server in the first queue (μ1).
        mu2: The service rate of the server in the second queue (μ2).
        capacity1: The capacity of the server in the first queue.
        capacity2: The capacity of the server in the second queue.
        q: The initial number of customers in the first queue.
        T: The simulation time horizon.
        N: The number of simulation runs to perform.
        log: A boolean flag indicating whether to print log messages during the simulation.
    """
    results_at_events = []
    results_at_time_intervals = []
    results = []
    for _ in range(N):
        env = simpy.Environment()
        (queue1, queue2) = setup(env=env, lambda_=lambda_, mu1=mu1, mu2=mu2, capacity1=capacity1, capacity2=capacity2, q=q, log=log)
        env.run(until=T)

        # Calculate time average number of customers
        total_customers = 0
        total_customers_at_events = (calculate_time_average(queue1.queue_length_at_events) + calculate_time_average(queue2.queue_length_at_events)) / T
        total_customers_at_time_intervals = (calculate_time_average(queue1.queue_length_over_time) + calculate_time_average(queue2.queue_length_over_time)) / T
        for time, length in queue1.queue_length_over_time + queue2.queue_length_over_time:
            total_customers += length
        time_average_customers = total_customers / env.now

        results_at_time_intervals.append(total_customers_at_time_intervals)
        results_at_events.append(total_customers_at_events)
        results.append(time_average_customers)

    print("############## RESULTS ##############")
    # Print results
    print(f"Lambda: {lambda_}, Mu1: {mu1}, Mu2: {mu2}, T: {T}, Q: {q}")
    # print(f"Average time average customers at time intervals: {np.mean(results_at_time_intervals):.2f}")
    # print(f"Average time average customers at events: {np.mean(results_at_events):.2f}")  # Format output
    print(f"Average time average customers: {np.mean(results):.2f}, Standard deviation: {np.std(results):.2f}")  # Format output
    print(f"Average customer using stationary distribution: {(lambda_/(mu1 - lambda_)) + (lambda_/(mu2 - lambda_))}")

    # Plot queue lengths for one run
    if q == 1000 and T == 2000:
        plt.figure(figsize=(10, 6))
        plt.plot([time for time, _ in queue1.queue_length_at_events], [length for _, length in queue1.queue_length_at_events], label="Queue 1")
        plt.plot([time for time, _ in queue2.queue_length_at_events], [length for _, length in queue2.queue_length_at_events], label="Queue 2")
        plt.xlabel("Time")
        plt.ylabel("Queue Length")
        plt.title(f"Queue Lengths over Time (One Simulation Run)(λ = {lambda_},μ1 = {mu1},μ2 = {mu2} ,q = {q}, T = {T})")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Test cases
    log = False
    # run(lambda_=1, mu1=4, mu2=4, capacity1=1, capacity2=1, q=0, T=10000, log=log)
    # run(lambda_=1, mu1=4, mu2=4, capacity1=1, capacity2=1, q=0, T=2000, log=log)
    # run(lambda_=1, mu1=4, mu2=4, capacity1=1, capacity2=1, q=1000, T=2000, log=log)

    for lambda_ in [1, 5]:
        for mu1 in [2, 4]:
            for mu2 in [3, 4]:
                for T in [10, 50, 100, 1000]:
                    run(lambda_=lambda_, mu1=mu1, mu2=mu2, capacity1=1, capacity2=1, q=0, T=T, log=log)
                    run(lambda_=lambda_, mu1=mu1, mu2=mu2, capacity1=1, capacity2=1, q=1000, T=2000, log=log)



