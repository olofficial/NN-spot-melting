import thermal_simulation.melting as ts
import neural_network.neural_network as nn

def main(): 
    a = nn.training_main(pixels_per_m = 600, generate_new_data=True, num_sequences=1000)
main()