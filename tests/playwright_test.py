import time

import pytest
from playwright.sync_api import Page

PORT = '8501'
BASE_URL = f'localhost:{PORT}'

pytestmark = pytest.mark.dashboard


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
    page.goto('http://localhost:8501/')

    # Enter vasprun location
    page.get_by_label('filename').click()
    page.get_by_label('filename').fill('short_simulation/vasprun.xml')
    page.get_by_label('filename').press('Enter')

    # Correct SuperCell
    page.get_by_label('supercell x').click()
    page.get_by_label('supercell x').fill('2')
    page.get_by_label('supercell x').press('Enter')

    # Enable RDFs
    page.get_by_role('tab', name='RDF plots').click()
    page.get_by_test_id('stCheckbox').locator('span').click()

    # change equilibration steps
    page.get_by_label('Equilibration Steps').click()
    page.get_by_label('Equilibration Steps').fill('100')
    page.get_by_label('Equilibration Steps').press('Enter')

    # check fullscreen
    page.get_by_role('button', name='View fullscreen').nth(2).click()
    page.get_by_role('button', name='Exit fullscreen').click()

    # These values should be presented on the page
    page.get_by_text('0.511089').click()
    page.get_by_text('40.7774').click()
    page.get_by_text('119.858').click()
    page.get_by_text('1.70636e-09').click()
    page.get_by_text('(8.5+/-0.7)e+12').click()
    page.get_by_text('2.45567e+28').click()
