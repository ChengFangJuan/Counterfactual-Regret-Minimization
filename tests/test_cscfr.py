
from sample_CFR.pokerstrategy import *
from sample_CFR.pokergames import *
from sample_CFR.poker_cfr import *

def near(val, expected, distance=0.0001):
    return val >= (expected - distance) and val <= (expected + distance)

print('')
print('')
print('Testing Chance Sampling (CS) CFR')
print('')
print('')
print('Computing NE for Leduc poker')
leduc = leduc_rules()

cfr = ChanceSamplingCFR(leduc)

iterations_per_block = 10
blocks = 1000
for block in range(blocks):
    print('Iterations: {0}'.format(block * iterations_per_block))
    cfr.run(iterations_per_block)
    result = cfr.profile.best_response()
    print('Best response EV: {0}'.format(result[1]))
    print('Total exploitability: {0}'.format(sum(result[1])))
print('Done!')
print('')