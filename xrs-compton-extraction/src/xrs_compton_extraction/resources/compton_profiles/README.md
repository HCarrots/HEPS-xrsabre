# Hartree–Fock Compton profiles

No scientific profile tables are bundled. The supported first implementation is
an optional runtime adapter to xraylib's Biggs total and partial profiles. Install
it with `python -m pip install "xrs-compton-extraction[profiles]"`.

The upstream table uses `p_z` in atomic units and contains profiles for Z=1–102,
shell occupancies, and binding energies. Every analysis must record the xraylib
version and the Biggs–Mendelsohn–Mann DOI. A redistribution license, orbital
mapping, unit convention, source version, and checksum must be reviewed before
any copy of the table is added to this directory.
