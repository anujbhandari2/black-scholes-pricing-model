from numpy import exp, sqrt, log
from scipy.stats import norm

class BlackScholes:
    """
    A class to implement Black Scholes Options Pricing Model. 
    This model is used to calculate theoretical prices of European style options
    and their associated Greeks.
    """
    def __init__(
        self,
        time_to_maturity: float, # time until option expires
        strike: float, # strike price of the option
        current_price: float, # current price of the underlying
        volatility: float, # annual volatility of the underlying
        interest_rate: float, # risk free interest rate
    ):
        """
        Initialize Black-Scholes model with option parameters.
        Parameters are all positive floating point numbers.
        """

        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):

        # local variables for better readability
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        # Calculate d1 and d2
        # d1 is the standardized distance from current price to strike price, adjusted for time value of money and volatility
        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
        ) / (volatility * sqrt(time_to_maturity))

        # d2 is d1 adjusted for volatility over time
        d2 = d1 - volatility * sqrt(time_to_maturity)

        # Calculate Call and Put prices
        # Call = S₀N(d₁) - Ke^(-rT)N(d₂)
        # S₀ is current price, K is strike, r is interest rate, T is time to maturity
        call_price = current_price * norm.cdf(d1) - (strike * exp(-interest_rate * time_to_maturity) * norm.cdf(d2))

        # Put = Ke^(-rT)N(-d₂) - S₀N(-d₁)
        put_price = (strike * exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)) - current_price * norm.cdf(-d1)

        # store rsults
        self.call_price = call_price
        self.put_price = put_price

        # Calculate Greeks
        # Delta
        self.call_delta = norm.cdf(d1)  # Delta for Call (ranges 0 to 1)
        self.put_delta = norm.cdf(d1) - 1  # Delta for Put (ranges -1 to 0)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (current_price * volatility * sqrt(time_to_maturity))  # Gamma (same for Call and Put)
        self.put_gamma = self.call_gamma

        # Theta (daily)
        self.call_theta = (
            -current_price * norm.pdf(d1) * volatility / (2 * sqrt(time_to_maturity)) -
            interest_rate * strike * exp(-interest_rate * time_to_maturity) * norm.cdf(d2)) / 365
        self.put_theta = (
            -current_price * norm.pdf(d1) * volatility / (2 * sqrt(time_to_maturity)) +
            interest_rate * strike * exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)) / 365 

        # Vega (per 1% change in volatility)
        self.vega = (current_price * sqrt(time_to_maturity) * norm.pdf(d1)) / 100 

        # Rho (per 1% change in interest rate)
        self.call_rho = (strike * time_to_maturity * exp(-interest_rate * time_to_maturity) * norm.cdf(d2)) / 100 
        self.put_rho = (-strike * time_to_maturity * exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)) / 100

        return call_price, put_price

if __name__ == "__main__":
    # Example usage
    time_to_maturity = 1.0
    strike = 100.0
    current_price = 100.0
    volatility = 0.2
    interest_rate = 0.05

    # create model and run
    bs_model = BlackScholes(
        time_to_maturity=time_to_maturity,
        strike=strike,
        current_price=current_price,
        volatility=volatility,
        interest_rate=interest_rate
    )
    call_price, put_price = bs_model.calculate_prices()

    # print results
    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")
    print(f"Call Delta: {bs_model.call_delta:.4f}")
    print(f"Put Delta: {bs_model.put_delta:.4f}")
    print(f"Call and Put Gamma: {bs_model.call_gamma:.4f}")
    print(f"Call Theta: ${bs_model.call_theta:.4f} per day")
    print(f"Put Theta: ${bs_model.put_theta:.4f} per day")
    print(f"Vega: ${bs_model.vega:.4f} per 1% volatility change")
    print(f"Call Rho: ${bs_model.call_rho:.4f} per 1% rate change")
    print(f"Put Rho: ${bs_model.put_rho:.4f} per 1% rate change")

