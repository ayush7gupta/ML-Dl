#Problem Statement: we dont know the state of the day(cold, hot, normal). we have got a teenager diary in which it is
# stated which drink(soda, hot choclate, iceCream) he had on what day. We also have the transition probability of the
# days, going from one state to other. we also have the probabilty of the drink which the teenager ios likely to have
# on a given state. Our task is to tell the sequence of what day appear in a sequence.

# The solution provided here uses viterbi algorithm to compute the probabilities and compute the max for each day.
# Finally a sequence of states is the resultant output.
# states
start = -1;
cold = 0;
normal = 1;
hot = 2;
stateCount = 3
stateNames = ["cold", "normal", "hot"]

# outputs
hotChoc = 0;
soda = 1;
iceCream = 2

timeSteps = 7

# state transition probabilities
trans = {}
trans[(start, cold)] = .1
trans[(start, normal)] = .8
trans[(start, hot)] = .1

trans[(cold, cold)] = .7
trans[(cold, normal)] = .1
trans[(cold, hot)] = .2

trans[(normal, cold)] = .3
trans[(normal, normal)] = .4
trans[(normal, hot)] = .3

trans[(hot, cold)] = .2
trans[(hot, normal)] = .4
trans[(hot, hot)] = .4

# state outputs
output = {}
output[(cold, hotChoc)] = .7
output[(cold, soda)] = .3
output[(cold, iceCream)] = 0

output[(normal, hotChoc)] = .1
output[(normal, soda)] = .7
output[(normal, iceCream)] = .2

output[(hot, hotChoc)] = 0
output[(hot, soda)] = .6
output[(hot, iceCream)] = .4

diary = [soda, soda, hotChoc, iceCream, soda, soda, iceCream]

# manage cell values and back pointers
cells = {}
backStates = {}


def computeMaxPrev(t, sNext):
    maxValue = 0
    maxState = 0

    for s in range(stateCount):
        value = cells[t, s] * trans[(s, sNext)]
        if (s == 0 or value > maxValue):
            maxValue = value
            maxState = s

    return (maxValue, maxState)


def viterbi(trans, output, diary):
    # special handling for t=0 which have no prior states)
    for s in range(stateCount):
        cells[(0, s)] = trans[(start, s)] * output[(s, diary[0])]

    # handle rest of time steps
    for t in range(1, timeSteps):
        for s in range(stateCount):
            maxValue, maxState = computeMaxPrev(t - 1, s)
            backStates[(t, s)] = maxState
            cells[(t, s)] = maxValue * output[(s, diary[t])]
            # print("t=", t, "s=", s, "maxValue=", maxValue, "maxState=", maxState, "output=", output[(s, diary[t])], "equals=", cells[(t, s)])

    # walk thru cells backwards to get most probable path
    path = []

    for tt in range(timeSteps):
        t = timeSteps - tt - 1  # step t backwards over timesteps
        maxValue = 0
        maxState = 0

        for s in range(stateCount):
            value = cells[t, s]
            if (s == 0 or value > maxValue):
                maxValue = value
                maxState = s

        path.insert(0, maxState)

    return path


# test our algorithm on the weather problem
path = viterbi(trans, output, diary)

print("Weather by days:")
for i in range(timeSteps):
    state = path[i]
    print("  day=", i + 1, stateNames[state])