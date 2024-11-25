import autode as ade
if __name__ == "__main__":  
	ade.Config.n_cores=24
	ade.Config.max_core=4000
	ade.Config.hcode="G16"
	ade.Config.lcode ="xtb"
	rxn=ade.Reaction(r"COC(=O)C(C=CO)=COC(C)=O.O=C1C=CC(=O)c2c(O)cccc21>>COC(=O)C1=C[C@@H](O)[C@@H]2C(=O)c3cccc(O)c3C(=O)[C@@H]2[C@@H]1OC(C)=O")
	ade.Config.G16.keywords.set_functional('cam-b3lyp')
	ade.Config.G16.keywords.opt.basis_set = '6-311++G**' 
	ade.Config.G16.keywords.opt_ts.basis_set = '6-311++G**' 
	ade.Config.G16.keywords.hess.basis_set = '6-311++G**' 
	ade.Config.G16.keywords.low_opt.basis_set = '6-31G*' 
	ade.Config.G16.keywords.low_opt.max_opt_cycles = 15
	ade.Config.num_conformers=1000
	ade.Config.rmsd_threshold=0.1
	ade.Config.hmethod_conformers=True
	rxn.calculate_reaction_profile(free_energy=True)
	for reac in rxn.reacs:
		if reac.imaginary_frequencies != None:
			print(f"{reac.name} has an imaginary frequency")
	for prod in rxn.prods:
		if prod.imaginary_frequencies != None:
			print(f"{prod.name} has an imaginary frequency")
