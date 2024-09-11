import re


def is_valid_edoal(txt):
    return txt.endswith('</rdf:RDF>')


def can_repair(txt):
    return txt.rfind('<map>') > 0


def merge_edoals(outputs):
    repaired_edoals = []
    for output in outputs:

        if not output.startswith('<?xml version'):
            output = '''<?xml version='1.0' encoding='utf-8' standalone='no'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
         xmlns:xsd='http://www.w3.org/2001/XMLSchema#'
         xmlns:alext='http://exmo.inrialpes.fr/align/ext/1.0/'
         xmlns:align='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
         xmlns:edoal='http://ns.inria.org/edoal/1.0/#'>\n''' + output

        output = re.sub(r'<Ontology rdf:about="([^"]+)" />',
                        r'<Ontology rdf:about="\1"><location>\1</location><formalism><Formalism align:name="owl" align:uri="http://www.w3.org/TR/owl-guide/"/></formalism></Ontology>',
                        output)
        if not is_valid_edoal(output) and can_repair(output):
            last_map_index = output.rfind('<map>')
            repaired_edoals.append(output[:last_map_index] + '\n\t</Alignment>\n</rdf:RDF>')
        else:
            repaired_edoals.append(output)

    final_edoal = None
    if len(repaired_edoals) > 1:
        final_edoal = ''
        first = repaired_edoals[0]
        final_edoal += first[:first.find('<map>')]
        for e in repaired_edoals[1:]:
            final_edoal += e[e.find('<map>'):e.rfind('</map>')] + '\n\t</map>'

        final_edoal += '\n\t</Alignment>\n</rdf:RDF>'

    elif len(repaired_edoals) == 1:
        final_edoal = repaired_edoals[0]

    return final_edoal
