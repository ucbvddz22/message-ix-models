from ixmp.reporting import Key


#: Replacements used in :meth:`collapse`.
REPLACE = {
    # Applied to whole values along each dimension
    'c': {
        'Crudeoil': 'Oil',
        'Electr': 'Electricity',
        'Ethanol': 'Liquids|Biomass',
        'Lightoil': 'Liquids|Oil',
    },
    'l': {
        'Final Energy': 'Final Energy|Residential',
    },
    't': {
        'Ch4 Rescomm': 'Energy|Demand|Residential and Commercial',
        'Ch4 Transport':
            'Energy|Demand|Transportation|Road Rail and Domestic Shipping',
    },

    # Applied after the variable column is assembled. Partial string
    # replacement; handled as regular expressions.
    'variable': {
        r'(Emissions\|CH4[^\|]*)\|((Gases|Liquids|Solids|Elec|Heat).*)$':
            r'\1|Energy|Supply|\2|Fugitive',

        r'Residential\|(Biomass|Coal)': r'Residential|Solids|\1',
        r'Residential\|Gas': 'Residential|Gases|Natural Gas',
        r"Import Energy\|Lng": "Primary Energy|Gas",
        r"Import Energy\|Coal": "Primary Energy|Coal",
        r"Import Energy\|Oil": "Primary Energy|Oil",
        r"Import Energy\|(Liquids|Oil)": r"Secondary Energy|\1",
        r"Import Energy\|(Liquids|Biomass)": r"Secondary Energy|\1",
        r"Import Energy\|Lh2": "Secondary Energy|Hydrogen",

    }
}


def collapse(df, var_name, var=[], region=[], replace_common=True):
    """Callback for the `collapse` argument to :meth:`.convert_pyam`.

    Simplified from :meth:`message_ix.reporting.pyam.collapse_message_cols`.

    The dimensions listed in the `var` and `region` arguments are automatically
    dropped from the returned :class:`.IamDataFrame`.

    Parameters
    ----------
    var_name : str
        Initial value to populate the IAMC 'Variable' column.
    var : list of str, optional
        Dimensions to concatenate to the 'Variable' column. These are joined
        after the `var_name` using the pipe ('|') character.
    region : list of str, optional
        Dimensions to concatenate to the 'Region' column.
    replace_common : bool, optional
        If :obj:`True` (the default), use :data`REPLACE` to perform standard
        replacements on columns before and after assembling the 'Variable'
        column.

    See also
    --------
    .core.add_iamc_table
    """
    if replace_common:
        for dim in 'clt':
            if dim in df.columns:
                df[dim] = df[dim].astype(str).str.title()

        try:
            # Level: to title case, add the word 'energy'
            # FIXME astype() here should not be necessary; debug
            df['l'] = df['l'].astype(str).str.title() + ' Energy'
        except KeyError:
            pass
        try:
            # Commodity: to title case
            # FIXME astype() here should not be necessary; debug
            df['c'] = df['c'].astype(str).str.title()
        except KeyError:
            pass

        # Apply replacements
        df = df.replace(REPLACE)

    # Extend region column ('n' and 'nl' are automatically added by message_ix)
    df['region'] = df['region'].astype(str) \
                               .str.cat([df[c] for c in region], sep='|')

    # Assemble variable column
    df['variable'] = var_name
    df['variable'] = df['variable'].str.cat([df[c] for c in var], sep='|')

    # TODO roll this into the rename_vars argument of message_ix...as_pyam()
    if replace_common:
        # Apply variable name partial replacements
        for pat, repl in REPLACE['variable'].items():
            df['variable'] = df['variable'].str.replace(pat, repl)

    # Drop same columns
    return df.drop(var + region, axis=1)


def infer_keys(reporter, key_or_keys, dims=[]):
    """Helper to guess complete keys in *reporter*."""
    single = isinstance(key_or_keys, (str, Key))
    keys = [key_or_keys] if single else key_or_keys

    result = []

    for k in keys:
        # Has some dimensions or tag
        key = Key.from_str_or_key(k) if ':' in k else k

        if '::' in k or key not in reporter:
            key = reporter.full_key(key)

        if dims:
            # Drop all but *dims*
            key = key.drop(*[d for d in key.dims if d not in dims])

        result.append(key)

    return result[0] if single else result
