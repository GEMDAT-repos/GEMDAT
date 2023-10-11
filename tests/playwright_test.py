import time

import pytest
from playwright.sync_api import Page, expect

PORT = '8501'
BASE_URL = f'http://localhost:{PORT}'

if pytest.skip_dashboard:
    pytestmark = pytest.mark.skip(
        reason='Use `--dashboard` to test dashboard workflow.')


@pytest.fixture(scope='module', autouse=True)
def before_module():
    """Run dashboard in module scope."""
    import subprocess

    p = subprocess.Popen([
        'gemdash',
        '--server.port',
        PORT,
        '--server.headless',
        'true',
    ])
    time.sleep(5)
    yield
    p.kill()


def test_gemdash(page: Page):
    # Goto web page
    page.goto(BASE_URL)

    # Enter vasprun location
    page.get_by_label('filename').click()
    page.get_by_label('filename').fill(
        'tests/data/short_simulation/vasprun.xml')
    page.get_by_label('filename').press('Enter')

    # this will take a wile, increase timeout on next check

    # Correct SuperCell
    expect(page.get_by_label('supercell x')).to_be_visible(timeout=100_000)
    page.get_by_label('supercell x').click()
    page.get_by_label('supercell x').fill('2')
    page.get_by_label('supercell x').press('Enter')

    # Check if pictures are present
    expect(page.locator('img').first).to_be_visible()

    # Enable RDFs
    page.get_by_role('tab', name='RDF plots').click()
    page.get_by_test_id('stCheckbox').locator('span').click()

    # Check pictures
    expect(page.get_by_role('img',
                            name='0').first).to_be_visible(timeout=10_000)
    expect(page.get_by_role('img', name='0').nth(1)).to_be_visible()
    expect(page.get_by_role('img', name='0').nth(2)).to_be_visible()

    # change equilibration steps
    page.get_by_label('Equilibration Steps').click()
    page.get_by_label('Equilibration Steps').fill('100')
    page.get_by_label('Equilibration Steps').press('Enter')

    # check fullscreen

    # These values should be presented on the page
    expect(page.get_by_text('0.511089')).to_be_visible(timeout=10_000)
    expect(page.get_by_text('40.7774')).to_be_visible()
    expect(page.get_by_text('119.858')).to_be_visible()
    expect(page.get_by_text('1.70636e-09')).to_be_visible()
    expect(page.get_by_text('(8.5+/-0.7)e+12')).to_be_visible()
    expect(page.get_by_text('2.45567e+28')).to_be_visible()
