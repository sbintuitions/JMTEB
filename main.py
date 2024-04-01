from jmteb.main import main, parse_args

if __name__ == "__main__":
    args = parse_args()

    main(
        text_embedder=args.embedder,
        evaluators=args.evaluators,
        save_dir=args.save_dir,
        overwrite_cache=args.overwrite_cache,
    )
