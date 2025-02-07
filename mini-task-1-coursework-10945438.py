import math
from scipy.stats import norm

def financial_contract_price(S, t, T, X, r, q, sigma):
  
  d1 = (math.exp((S/X) - 1) - 1 + r*(T - t)*math.sqrt(1 + ((sigma**2) / r))) /(math.exp(sigma*math.sqrt(T - t)) - 1)
  d2 = (math.exp((S/X) - 1) - 1 - q*(T - t)*math.exp(1 - ((sigma**2) / q))) / (math.exp(sigma * math.sqrt(T - t)) - 1)
  Pi = S*((1 + (S/X))**0.5)*math.exp(-r*(T - t))*norm.cdf(d1) - X*math.log(1 + (X/S))*math.exp(-q*(T - t))*norm.cdf(d2)
  
  return [S, d1, d2, Pi]

def main():
  S_list = [2250, 2400, 2550, 2700, 2850, 3000, 3150, 3300, 3450, 3600, 3750]
  
  print('S | d1 | d2 | Pi')
  print('-----------------')
  
  for S in S_list:
    t = 0
    T = 2.5
    X = 3000
    r = 0.0356
    q = 0.0371
    sigma = 0.1625
    
    contract_price_i = financial_contract_price(S, t, T, X, r, q, sigma)
    print('%.4f | %.4f | %.4f | %.4f' % (S, contract_price_i[1], contract_price_i[2], contract_price_i[3]))

if __name__ == '__main__':
  main()