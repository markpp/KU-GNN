from callbacks import Step

#lr_reduction = 5

def onetenth_4_8_12(lr):
    steps = [4, 8, 12]
    lrs = [lr, lr / 10, lr / 100, lr / 1000]
    return Step(steps, lrs)

def onetenth_10_15_20(lr):
    steps = [10, 15, 15]
    lrs = [lr, lr / 10, lr / 100, lr / 1000]
    return Step(steps, lrs)

def onetenth_30_45(lr):
    steps = [30, 45]
    lrs = [lr, lr / 10, lr / 100]
    return Step(steps, lrs)

def onetenth_50_75(lr):
    steps = [50, 75]
    lrs = [lr, lr / 10, lr / 100]
    return Step(steps, lrs)

def onetenth_75_95(lr):
    steps = [75, 95]
    lrs = [lr, lr / 10, lr / 100]
    return Step(steps, lrs)


def onetenth_75_100(lr):
    steps = [75, 100]
    lrs = [lr, lr / 10, lr / 100]
    return Step(steps, lrs)

def onetenth_150_175(lr):
    steps = [150, 175]
    lrs = [lr, lr / 5, lr / 50]
    return Step(steps, lrs)

def onetenth_150_200(lr):
    steps = [150, 200]
    lrs = [lr, lr / 10, lr / 100]
    return Step(steps, lrs)

def onetenth_150_200_250(lr):
    steps = [150, 20, 250]
    lrs = [lr, lr / 10, lr / 100, lr / 500]
    return Step(steps, lrs)

def wideresnet_step(lr):
    steps = [60, 120, 160]
    lrs = [lr, lr / 5, lr / 25, lr / 125]
    return Step(steps, lrs)
