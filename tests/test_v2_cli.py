from causaflux.cli import build_parser

def test_v2_cli_commands_registered():
    parser=build_parser()
    for argv in [
        ['v2-run','--output','x'],
        ['v2-validate','--input','x'],
        ['longitudinal-contract','--output','x'],
        ['longitudinal-convert','--input','a.csv','--output','a.npz'],
        ['longitudinal-benchmark','--input','a.csv'],
        ['shift-calibration','--input','a.csv'],
    ]:
        args=parser.parse_args(argv)
        assert callable(args.func)
