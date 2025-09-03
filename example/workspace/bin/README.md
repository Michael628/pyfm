This bin/ directory should hold any executables required for eaach job step to be used in the params.yaml file. For example:

- HadronsMILC for staggered Hadrons calculations: <https://github.com/Michael628/HadronsMILC>
- make_links_hisq to smear using milc: <https://github.com/milc-qcd/milc_qcd/blob/f816b1789ea7e519a3f99bb844441442a3438908/ks_imp_utilities/Make_template#L147>

Note: The contraction step is handled by the pyfm.a2a module.
