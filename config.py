#Parameters

DST1max = 15
DST2max = 15
DST3max = 15
DST4max = 15
DST5max = 15
DST6max = 15
DST7max = 15
DST8max = 15

DSD1max = 15
DSD2max = 15
DSD3max = 15
DSD4max = 15

Bucket_Anglemax = 95
Dipper_Anglemax = 145
Boom_Anglemax = 70
Slew_Anglemax = 180

# min

DST1min = 0
DST2min = 0
DST3min = 0
DST4min = 0
DST5min = 0
DST6min = 0
DST7min = 0
DST8min = 0

DSD1min = 0
DSD2min = 0
DSD3min = 0
DSD4min = 0

Bucket_Anglemin = -50
Dipper_Anglemin = 30
Boom_Anglemin = -20
Slew_Anglemin = -180

d = 0

# Training Loop
total_numsteps = 0
n_episodes = 10

best_score = 0
score_history = []
early_stop = False
request = 0
stop_command = 0
best_avg = 1
max_episodes = 10
max_steps = 100000000

reward_total = list()
rewards = []

action = [0]


