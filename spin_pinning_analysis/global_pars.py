# commonly used variables
blowvar = 'LocalVariables.ImagingBlowOutID'
blowdetvar = 'GlobalImagingVariables2.SecondSpinBlowOutPulseMagneticField' # 610 means doublons
intvar = 'GlobalLatticeVariables2.PhysicsSplitRadialLoadingField'
patvar = 'GlobalLatticeVariables2.Dmd0SecondPatternFilename'
bivar = 'GlobalLatticeVariables2.IsImageBIFormation'
xbar = 'LocalVariables.XbarFrequency'
densvar = 'GlobalLatticeVariables2.ThirdDMD1RampPower'
tempvar = 'GlobalLatticeVariables2.LoadingGradientLatticeHoldingTime'


# parameters about centering CHECK THESE!!
# shift_center = (1,1)
yxCentersite = (82,79)
# # default
# AACrop = (5,-4,5,-4)
# nx = 81
# ny = 81
# center = [40+shift_center[0],40+shift_center[1]]
# # more cropped - 20 less on all sides
# AACrop = (25,-24,25,-24)
# nx = 41
# ny = 41
# center = [20+shift_center[0],20+shift_center[1]]
# # more cropped - 15 less on all sides
# AACrop = (20,-19,20,-19)
# nx = 51
# ny = 51
# center = [25+shift_center[0],25+shift_center[1]]
# # more cropped - 20 less on all sides, diff center
AACrop = (27,-22,26,-23)
nx = 41
ny = 41
# center = [18+shift_center[0],19+shift_center[1]]
center = [18,19]


# postselection params : 3 is doublon
num_low = {0: 300, 1: 150, 2: 150, 3: 10}
num_high = {0: 500, 1: 250, 2: 250, 3: 50}

# weighting
p_avg_weight = 1/3

