# Project 3 - 2D Square Lattice XY Model

*For the final project of Computational Physics we encourage you to define your own project!*

*The only requirement we ask for is that it should be a simulation of a physics-related project (but we consider physics-related rather broad, and e.g. simulation of biological systems or an engineering problem are also absolutely ok).*
*We do advise you though to start from a project where there is some existing literature. in this way you have a starting point as well as something to compare your simulation to for validation.*

*We encourage you to discuss your project idea with us during the time of class, or remotely via the planning issue #1.*

*In any case, you need to fill in a short plan (a few lines) together with a reference to literature in the planning issue, and have it agreed by us before May 27 (i.e. latest two weeks before the presentation).*

*If you have problems to come up with a good project, we can provide you with proven project ideas. But we first want you to try coming up with your own project! Having designed your own project will also give a small bonus for the grade.*

## Authors

* @[kwo](https://gitlab.kwant-project.org/kwo)
* @[juandaanieel](https://gitlab.kwant-project.org/juandaanieel)

## Links to issues

* [Issue 1](https://gitlab.kwant-project.org/computational_physics/projects/Project-3_kwo/-/issues/1)
* [Issue 2](https://gitlab.kwant-project.org/computational_physics/projects/Project-3_kwo/-/issues/2)
* [Issue 3](https://gitlab.kwant-project.org/computational_physics/projects/Project-3_kwo/-/issues/3)

## Journal: Week 1

This week, we have implemented the high-speed Metropolis algorithm using `NumPy` and `Numba`. The evolution of the simulation can be seen in the animated `.webm` below, where we begin with a completely homogeneous lattice state, and slowly increase the temperature past the critical temperature $`T_c≈0.892937J/k_B`$, (see [here](http://www.lps.ens.fr/~krauth/images/7/72/Stage_Mayer_Johannes_2015.pdf))).

*Note: The $`T`$ in the plot title is in units of $`T_c`$. Thus, if it shows $`T`=1$, it means the system is at the critical temperature.*

![L=256, τ=10000000, τ_n=25, T_i=2Tc, T_f=0](simulation_images/Metropolis_L256_tau10000000_nrelax25_Tinit2_Tfinal0.webm)
<!-- <img src="" width="360" height="307" /> -->
where τ is the relaxation time, i.e. the number of metropolis iterations allocated for the system to equilibrate for a given temperature. $`\tau_n`$ is the number of temperature points between the initial temperature $`T_i`$ and the final temperature $`T_f`$.

Here, we see another animated `.webm`, where we begin with a completely random spin lattice, and immediately quench the system to $`T=0`$ with 100000000 iterations.

![L=256, τ=100000000, τ_n=1, T_i=0, T_f=0](simulation_images/Metropolis_L256_tau100000000_nrelax1_Tinit0_Tfinal0.webm)
