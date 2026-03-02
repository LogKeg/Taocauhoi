"""
OMML (Office Math Markup Language) to LaTeX converter.
"""


def omml_to_latex(element, ns: dict) -> str:
    """
    Convert an OMML math element to LaTeX notation.

    Handles common OMML elements:
    - m:f (fraction) -> \\frac{num}{den}
    - m:sSup (superscript) -> base^{exp}
    - m:sSub (subscript) -> base_{idx}
    - m:sSubSup (sub+superscript) -> base_{sub}^{sup}
    - m:rad (radical/sqrt) -> \\sqrt{content} or \\sqrt[n]{content}
    - m:nary (summation, product, integral) -> \\sum, \\prod, \\int
    - m:d (delimiter/parentheses) -> \\left( ... \\right)
    - m:t (text) -> plain text
    """
    if element is None:
        return ''
    if element.tag is None:
        return ''

    ns_prefix = ns.get('m', '')
    tag = element.tag.replace(ns_prefix, '').replace('{', '').replace('}', '')

    # Text element
    if tag == 't':
        return element.text or ''

    # Fraction
    if tag == 'f':
        num = element.find('m:num', ns)
        den = element.find('m:den', ns)
        num_latex = omml_children_to_latex(num, ns) if num is not None else ''
        den_latex = omml_children_to_latex(den, ns) if den is not None else ''
        return f'\\frac{{{num_latex}}}{{{den_latex}}}'

    # Superscript
    if tag == 'sSup':
        base = element.find('m:e', ns)
        sup = element.find('m:sup', ns)
        base_latex = omml_children_to_latex(base, ns) if base is not None else ''
        sup_latex = omml_children_to_latex(sup, ns) if sup is not None else ''
        return f'{base_latex}^{{{sup_latex}}}'

    # Subscript
    if tag == 'sSub':
        base = element.find('m:e', ns)
        sub = element.find('m:sub', ns)
        base_latex = omml_children_to_latex(base, ns) if base is not None else ''
        sub_latex = omml_children_to_latex(sub, ns) if sub is not None else ''
        return f'{base_latex}_{{{sub_latex}}}'

    # Subscript + Superscript
    if tag == 'sSubSup':
        base = element.find('m:e', ns)
        sub = element.find('m:sub', ns)
        sup = element.find('m:sup', ns)
        base_latex = omml_children_to_latex(base, ns) if base is not None else ''
        sub_latex = omml_children_to_latex(sub, ns) if sub is not None else ''
        sup_latex = omml_children_to_latex(sup, ns) if sup is not None else ''
        return f'{base_latex}_{{{sub_latex}}}^{{{sup_latex}}}'

    # Radical (square root)
    if tag == 'rad':
        deg = element.find('m:deg', ns)
        content = element.find('m:e', ns)
        content_latex = omml_children_to_latex(content, ns) if content is not None else ''
        deg_latex = omml_children_to_latex(deg, ns) if deg is not None else ''
        if deg_latex and deg_latex.strip():
            return f'\\sqrt[{deg_latex}]{{{content_latex}}}'
        return f'\\sqrt{{{content_latex}}}'

    # N-ary (sum, product, integral)
    if tag == 'nary':
        chr_elem = element.find('.//m:chr', ns)
        sub = element.find('m:sub', ns)
        sup = element.find('m:sup', ns)
        content = element.find('m:e', ns)

        # Get the operator character safely
        op_char = '∑'  # default
        if chr_elem is not None:
            ns_m = ns.get('m', '')
            op_char = chr_elem.get(f'{{{ns_m}}}val') or chr_elem.get('val') or '∑'
        op_map = {'∑': '\\sum', '∏': '\\prod', '∫': '\\int', '⋃': '\\bigcup', '⋂': '\\bigcap'}
        op_latex = op_map.get(op_char, op_char) or '\\sum'

        sub_latex = omml_children_to_latex(sub, ns) if sub is not None else ''
        sup_latex = omml_children_to_latex(sup, ns) if sup is not None else ''
        content_latex = omml_children_to_latex(content, ns) if content is not None else ''

        result = op_latex
        if sub_latex:
            result += f'_{{{sub_latex}}}'
        if sup_latex:
            result += f'^{{{sup_latex}}}'
        result += f' {content_latex}'
        return result

    # Delimiter (parentheses, brackets)
    if tag == 'd':
        content = element.find('m:e', ns)
        content_latex = omml_children_to_latex(content, ns) if content is not None else ''
        # Get delimiter characters safely
        ns_m = ns.get('m', '')
        beg_chr = element.find('.//m:begChr', ns)
        end_chr = element.find('.//m:endChr', ns)
        beg = '('
        end = ')'
        if beg_chr is not None:
            beg = beg_chr.get(f'{{{ns_m}}}val') or beg_chr.get('val') or '('
        if end_chr is not None:
            end = end_chr.get(f'{{{ns_m}}}val') or end_chr.get('val') or ')'
        return f'\\left{beg}{content_latex}\\right{end}'

    # Matrix
    if tag == 'm':
        rows = element.findall('m:mr', ns)
        row_latexes = []
        for row in rows:
            cells = row.findall('m:e', ns)
            cell_latexes = [omml_children_to_latex(c, ns) for c in cells]
            row_latexes.append(' & '.join(cell_latexes))
        return '\\begin{matrix}' + ' \\\\ '.join(row_latexes) + '\\end{matrix}'

    # Run element (container for text)
    if tag == 'r':
        return omml_children_to_latex(element, ns)

    # Default: recursively process children
    return omml_children_to_latex(element, ns)


def omml_children_to_latex(element, ns: dict) -> str:
    """Process all children of an OMML element and return LaTeX."""
    if element is None:
        return ''
    result = []
    for child in element:
        result.append(omml_to_latex(child, ns))
    return ''.join(result)
