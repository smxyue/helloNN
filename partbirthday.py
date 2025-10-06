import random
from datetime import date
PARTY_SIZE = 80
def check_shared_birthday():
    """
    Generate 23 random dates (month/day only) and check if at least two are the same.
    
    Returns:
        bool: True if at least two dates are the same, False otherwise
    """
    # Generate 23 random dates
    dates = []
    for _ in range(PARTY_SIZE):
        # Generate random month (1-12)
        month = random.randint(1, 12)
        
        # Generate appropriate random day based on month
        if month in [1, 3, 5, 7, 8, 10, 12]:  # 31 days
            day = random.randint(1, 31)
        elif month in [4, 6, 9, 11]:  # 30 days
            day = random.randint(1, 30)
        else:  # February - assuming 28 days for simplicity
            day = random.randint(1, 28)
        
        # Store as tuple (month, day)
        dates.append((month, day))
    
    # Check if there are duplicate dates
    return len(dates) != len(set(dates))

def simulate_birthday_paradox(n):
    """
    Run the birthday paradox simulation N times and calculate the probability.
    
    Args:
        n (int): Number of simulations to run
        
    Returns:
        tuple: (number of matches, percentage of matches)
    """
    matches = 0
    
    # Run the simulation N times
    for _ in range(n):
        if check_shared_birthday():
            matches += 1
    
    # Calculate percentage
    percentage = (matches / n) * 100
    
    return matches, percentage

# Example usage:
if __name__ == "__main__":
    # Run simulation 1000 times
    n_simulations = 100000
    matches, percentage = simulate_birthday_paradox(n_simulations)
    
    print(f"Number of simulations: {n_simulations}")
    print(f"Number of matches: {matches}")
    print(f"Percentage: {percentage:.3f}%")