from pycirculate.anova import AnovaController

# Your device's MAC address can be found with `sudo hcitool lescan`
anova = AnovaController("00:81:F9:D8:3F:94")

anova.read_unit()
# -> 'c'
anova.read_temp()
# -> '14.9'

anova.set_temp(63.5)
anova.start_anova()

anova.anova_status()
# -> 'running'