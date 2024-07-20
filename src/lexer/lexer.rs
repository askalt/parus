use crate::grammar::grammar::GrammarSymbol;

/// Describes a lexer over some grammar.
pub trait Lexer<S>
where
    S: GrammarSymbol,
{
    /// Returns next symbol, if one exists.
    /// Else returns `None`.
    fn next(&mut self) -> Option<S> {
        let ret = self.cur();
        if matches!(ret, Some(_)) {
            self.shift();
        }
        ret
    }

    /// Returns current symbol, if one exists.
    /// Else returns `None`.
    fn cur(&self) -> Option<S>;

    /// Shifts current position by 1 to the right.
    fn shift(&mut self);
}
