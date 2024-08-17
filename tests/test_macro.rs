use grammar_derive::GrammarSymbol;

fn i32_generator() -> i32 {
    42
}

// TODO:
// * custom implementation for specific symbol.

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug, GrammarSymbol)]
enum Symbol {
    // 0 is always start non terminal.
    #[to(
        A B Int,
        Epsilon,
    )]
    S,

    A,
    B,
    #[param_gen(i32_generator)]
    Int(i32),
}
