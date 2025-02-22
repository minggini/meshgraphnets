#!/bin/bash
#SBATCH --job-name=ripple_gradient_sum_noattention
#SBATCH --partition=gpu_8

for i in 7
do
	srun --exclusive -N1 -p gpu_8 --gres=gpu python run_model.py --message_passing_aggregator=sum --ripple_used=True --ripple_generation=gradient --ripple_generation_number=4 --ripple_node_selection=random --ripple_node_selection_random_top_n=10 --ripple_node_connection=most_influential --message_passing_steps=${i} --attention=False --epochs=25 --trajectories=1000 --num_rollouts=100
done
