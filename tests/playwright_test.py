from __future__ import annotations

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

    page.get_by_label('filename').click()
    page.get_by_label('filename').fill(
        'tests/data/short_simulation/vasprun.xml')
    page.get_by_label('filename').press('Enter')

    # check values
    expect(page.get_by_text('1.57061e-09')).to_be_visible(timeout=60_000)
    expect(page.get_by_text('110.322')).to_be_visible()
    expect(page.get_by_text('2.45567e+28')).to_be_visible()
    expect(page.get_by_text('(8.5+/-0.9)e+12')).to_be_visible()
    expect(page.get_by_text('0.52043')).to_be_visible()
    expect(page.get_by_text('40.7774')).to_be_visible()

    # check plots
    expect(page.get_by_role('img', name='0').nth(1)).to_be_visible()
    expect(page.get_by_role('img', name='0').nth(2)).to_be_visible()
    expect(page.get_by_role('img', name='0').first).to_be_visible()
    expect(
        page.get_by_text(
            '02460510152025303540ElementLiPSBrDisplacement per elementDisplacement (Angstrom)'
        )).to_be_visible()
    expect(
        page.get_by_text(
            '010002000300001234Br + stdS + stdP + stdLi + stdDisplacement per elementTime ste'
        )).to_be_visible()

    page.locator('label').filter(
        has_text='Error Analysis').locator('span').click()

    expect(
        page.get_by_text(
            '024051015202530ElementLiPSBrDisplacement per element after 374 timestepsDisplace'
        )).to_be_visible(timeout=10_000)

    page.get_by_role('tab', name='RDF plots').click()
    page.locator('label').filter(has_text='Plot RDFs').locator('span').click()

    expect(page.get_by_role('img',
                            name='0').first).to_be_visible(timeout=60_000)
    expect(page.get_by_role('img', name='0').nth(1)).to_be_visible()
    expect(page.get_by_role('img', name='0').nth(2)).to_be_visible()

    page.get_by_role('tab', name='Density plots').click()
    page.get_by_text('Generate density').click()

    expect(page.get_by_label('Density plots')).to_be_visible(timeout=60_000)
