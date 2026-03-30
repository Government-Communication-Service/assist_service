"""
Script to scrape the GOV.UK Style Guide A-Z page and extract all rules with their details.
Output is saved as JSON.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


def scrape_style_guide() -> List[Dict[str, str]]:
    """
    Scrape the GOV.UK style guide A-Z page and extract all rules.

    Returns:
        List of dictionaries containing 'rule' and 'details' for each entry.
    """
    url = "https://www.gov.uk/guidance/style-guide/a-to-z"

    # Fetch the page
    response = requests.get(url)
    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    rules = []

    # Non-content sections to skip
    skip_sections = {
        'Services and information', 'Government activity', 'Support links',
        'Before you suggest a change', 'How to suggest a change', 'What happens next',
        'About the A to Z', 'Suggest a change or addition', 'Search', 'Search GOV.UK'
    }

    # Structure: Look for h3 headings as rules with following paragraphs as details
    h3_elements = soup.find_all('h3')

    for h3 in h3_elements:
        rule = h3.get_text(strip=True)

        # Skip navigation/meta sections
        if rule in skip_sections:
            continue

        # Get the next sibling(s) until we hit another h3 or section boundary
        details_parts = []
        current = h3.find_next_sibling()

        while current and current.name not in ['h2', 'h3', 'h4']:
            if current.name in ['p', 'ul', 'ol', 'div']:
                text = current.get_text(strip=True)
                if text:
                    details_parts.append(text)
            current = current.find_next_sibling()

        if details_parts:
            details = ' '.join(details_parts)
            rule = clean_text(rule)
            details = clean_text(details)

            if rule and details:
                rules.append({
                    'rule': rule,
                    'details': details
                })

    return rules


def clean_text(text: str) -> str:
    """
    Clean up extracted text by removing extra whitespace.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def save_to_json(rules: List[Dict[str, str]], output_file: Path | None = None):
    """
    Save rules to a JSON file.

    Args:
        rules: List of rule dictionaries
        output_file: Output file path (defaults to web_scraper/output/gov_uk_style_guide.json)
    """
    if output_file is None:
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'gov_uk_style_guide.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(rules)} rules to {output_file}")


def main():
    """Main execution function."""
    print("Scraping GOV.UK Style Guide A-Z page...")

    try:
        rules = scrape_style_guide()

        if not rules:
            print("Warning: No rules found. The page structure may have changed.")
            return

        # Save to JSON
        save_to_json(rules)

        # Print some examples
        print("\nFirst 5 rules:")
        for i, rule in enumerate(rules[:5], 1):
            print(f"\n{i}. Rule: {rule['rule']}")
            print(f"   Details: {rule['details'][:100]}{'...' if len(rule['details']) > 100 else ''}")

    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
    except Exception as e:
        print(f"Error processing the page: {e}")
        raise


if __name__ == '__main__':
    main()
