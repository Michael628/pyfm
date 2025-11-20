#!/usr/bin/env python3
"""
Compare two XML files and show their differences.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from difflib import unified_diff
from pathlib import Path


def normalize_element(elem):
    """Normalize an XML element for comparison (sorts children and attributes)."""
    # Sort attributes
    if elem.attrib:
        elem.attrib = dict(sorted(elem.attrib.items()))

    # Normalize text (strip whitespace for comparison)
    if elem.text:
        elem.text = elem.text.strip()
    if elem.tail:
        elem.tail = elem.tail.strip()

    # Recursively normalize children
    for child in elem:
        normalize_element(child)

    # Sort children by tag name and text content to ensure consistent ordering
    # This makes the comparison order-independent
    elem[:] = sorted(elem, key=lambda e: (
        e.tag,
        e.text or '',
        str(sorted(e.attrib.items())) if e.attrib else '',
        ET.tostring(e, encoding='unicode')
    ))


def elements_equal(e1, e2, path=""):
    """
    Recursively compare two XML elements for semantic equality.
    Returns (is_equal, differences_list)
    """
    differences = []

    # Compare tags
    if e1.tag != e2.tag:
        differences.append(f"{path}: Different tags: '{e1.tag}' vs '{e2.tag}'")
        return False, differences

    # Compare text content
    text1 = (e1.text or '').strip()
    text2 = (e2.text or '').strip()
    if text1 != text2:
        differences.append(f"{path}/{e1.tag}: Different text content: '{text1}' vs '{text2}'")

    # Compare attributes
    if e1.attrib != e2.attrib:
        # Find specific attribute differences
        keys1 = set(e1.attrib.keys())
        keys2 = set(e2.attrib.keys())

        for key in keys1 - keys2:
            differences.append(f"{path}/{e1.tag}[@{key}]: Attribute only in first file: {e1.attrib[key]}")
        for key in keys2 - keys1:
            differences.append(f"{path}/{e1.tag}[@{key}]: Attribute only in second file: {e2.attrib[key]}")
        for key in keys1 & keys2:
            if e1.attrib[key] != e2.attrib[key]:
                differences.append(f"{path}/{e1.tag}[@{key}]: Different values: '{e1.attrib[key]}' vs '{e2.attrib[key]}'")

    # Compare children
    if len(e1) != len(e2):
        differences.append(f"{path}/{e1.tag}: Different number of children: {len(e1)} vs {len(e2)}")
        return False, differences

    # Compare each child (after normalization, they should be in the same order)
    for i, (child1, child2) in enumerate(zip(e1, e2)):
        child_equal, child_diffs = elements_equal(child1, child2, f"{path}/{e1.tag}[{i}]")
        differences.extend(child_diffs)

    return len(differences) == 0, differences


def compare_xml_files(file1, file2):
    """Compare two XML files semantically (ignoring element order)."""
    try:
        # Parse both XML files
        tree1 = ET.parse(file1)
        tree2 = ET.parse(file2)
        root1 = tree1.getroot()
        root2 = tree2.getroot()

        # Normalize both trees (sorts children and attributes)
        normalize_element(root1)
        normalize_element(root2)

        # Compare the normalized trees
        are_equal, differences = elements_equal(root1, root2)

        if are_equal:
            print(f"No differences found between {file1} and {file2}")
            return True
        else:
            print(f"Found {len(differences)} difference(s) between {file1} and {file2}:\n")
            for diff in differences:
                print(f"  {diff}")
            return False

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Compare two XML files and show their differences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s file1.xml file2.xml
  %(prog)s path/to/first.xml path/to/second.xml
        '''
    )

    parser.add_argument(
        'file1',
        type=Path,
        help='First XML file to compare'
    )

    parser.add_argument(
        'file2',
        type=Path,
        help='Second XML file to compare'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Comparing {args.file1} with {args.file2}...")

    # Compare the files
    are_identical = compare_xml_files(args.file1, args.file2)

    # Exit with appropriate code
    sys.exit(0 if are_identical else 1)


if __name__ == '__main__':
    main()
