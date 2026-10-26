import pytest

from miniagent.sandbox import run_python

pytestmark = pytest.mark.integration


def test_stdout_is_captured():
    result = run_python("print('hello')")
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"


def test_exception_goes_to_stderr():
    result = run_python("1 / 0")
    assert result.exit_code != 0
    assert "ZeroDivisionError" in result.stderr


def test_infinite_loop_is_killed():
    result = run_python("while True: pass", timeout_s=3)
    assert result.timed_out


def test_tmp_is_writable():
    result = run_python("open('/tmp/x.txt', 'w').write('ok'); print('written')")
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "code",
    [
        "import urllib.request; urllib.request.urlopen('http://example.com', timeout=3)",
        "open('/home/sandbox/x.txt', 'w').write('x')",
        "open('/etc/hacked', 'w').write('x')",
    ],
    ids=["no-network", "read-only-fs", "not-root"],
)
def test_dangerous_actions_fail(code):
    result = run_python(code)
    assert result.exit_code != 0


def test_data_dir_is_mounted_read_only(tmp_path):
    (tmp_path / "input.csv").write_text("a,b\n1,2\n")

    result = run_python(
        "import pandas as pd\n"
        "print(pd.read_csv('/data/input.csv').shape)\n"
        "open('/data/new.txt', 'w')\n",
        data_dir=tmp_path,
    )

    assert "(1, 2)" in result.stdout
    assert "Read-only file system" in result.stderr
