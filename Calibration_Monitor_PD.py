

def Calibration(Vset=50,set_temp=25,margin=5):
    Vout=4 ## Use Voltage value coming from ADC
    step=0.001 #Degrees
    dt=0
    while True:
        # Checking the Trend
        if Vset-Vout<-margin:
            step=-abs(step)
        elif Vset-Vout>margin:
            step=abs(step)
        else: break
        # Changing Probe laser temperature until error is within margin
        dt+=step
        new_temp=set_temp+step
        # function_set_laser_temp(new_temp)
    return
