use rand::Rng;
use std::{collections::VecDeque, fmt::Debug, hash::Hash, marker::PhantomData};

/// Describes the symbols in the specific grammar.
pub trait GrammarSymbol: Eq + Hash + Ord + Clone + Debug {
    /// Returns true is symbol is a terminal.
    fn is_terminal(num: usize) -> bool;

    /// Returns true is symbol is a non-terminal.
    fn is_non_terminal(num: usize) -> bool {
        return !Self::is_terminal(num);
    }

    /// Returns a start non terminal for the grammar.
    fn start_non_terminal() -> usize;

    /// Get productions for the specific symbol.
    /// Will be called for only non-terminal symbols.
    /// `None` means epsilon-production.
    fn get_productions<'a, 'b, 'c>(num: usize) -> &'b [Option<&'c [usize]>];

    /// Make `Self` from symbol num.
    /// Only to pretty iterator results.
    /// Called only on non-terminals.
    fn from_num(num: usize) -> Self;

    /// Make symbol num from `Self`.
    fn to_num(&self) -> usize;
}

pub trait IterableGrammarSymbol: GrammarSymbol {
    /// Makes an iterator from the grammar.
    fn into_iterator() -> GrammarIterator<Self>
    where
        Self: Sized,
    {
        return GrammarIterator::new();
    }
}

/// Visits the specified grammar, using BFS.
/// If the grammar produces the same string several times, returns it several times.
pub struct GrammarIterator<S: GrammarSymbol> {
    queue: VecDeque<Vec<usize>>,
    phantom: PhantomData<S>,
}

impl<S: GrammarSymbol> GrammarIterator<S> {
    /// Creates new `GrammarIterator` over some grammar.
    fn new() -> Self {
        Self {
            queue: VecDeque::from([vec![S::start_non_terminal()]]),
            phantom: PhantomData,
        }
    }
}

/// Applies production to the string and returns the count of terminals.
fn apply_production<S: GrammarSymbol>(
    str: &mut Vec<usize>,
    i: usize,
    production: &Option<&[usize]>,
) -> i32 {
    if production.is_none() {
        // Epsilon production, so remove this symbol.
        str.remove(i);
        0
    } else {
        // Non-epsilon production, so replace subarray.
        let production = production.clone().unwrap();
        let mut count_non_terminals = 0;
        str.splice(
            i..i + 1,
            production
                .iter()
                .map(|it| {
                    if S::is_terminal(*it) {
                        count_non_terminals += 1;
                    }
                    it
                })
                .cloned(),
        );
        count_non_terminals
    }
}

impl<S: GrammarSymbol> Iterator for GrammarIterator<S> {
    type Item = Vec<S>;

    /// Return next derivate string.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(str) = self.queue.pop_front() {
            // Rewrite first non-terminal.
            let non_terminal = str
                .iter()
                .enumerate()
                .find(|(_, e)| S::is_non_terminal(**e));
            if non_terminal.is_none() {
                // Strings does not contain non-terminal => it in language.
                return Some(str.into_iter().map(|it| S::from_num(it)).collect());
            }
            let (i, s) = non_terminal.unwrap();
            for production in S::get_productions(*s) {
                let mut nxt = str.clone();
                apply_production::<S>(&mut nxt, i, production);
                self.queue.push_back(nxt);
            }
        }
        None
    }
}

pub struct RandomGrammarIterator<S: GrammarSymbol> {
    /// Tries not choose productions with only terminals while length < min_length.
    min_lentgth: usize,
    /// If exceeds max_length, tries to choose production without non-terminals.
    max_length: usize,
    phantom: PhantomData<S>,
    rng: rand::rngs::ThreadRng,
}

impl<S: GrammarSymbol> RandomGrammarIterator<S> {
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self {
            min_lentgth: min_length,
            max_length: max_length,
            phantom: PhantomData,
            rng: rand::thread_rng(),
        }
    }
}

/// Returns a random index preferring `important` indices set.
fn take_random_index(
    important: &Vec<usize>,
    non_important: &Vec<usize>,
    rng: &mut rand::rngs::ThreadRng,
) -> usize {
    assert!(!important.is_empty() || !non_important.is_empty());
    let id = rng.gen::<usize>();
    if important.is_empty() {
        non_important[id % non_important.len()]
    } else {
        important[id % important.len()]
    }
}

/// Returns random strings over grammar. Never return None.
impl<S: GrammarSymbol> Iterator for RandomGrammarIterator<S> {
    type Item = Vec<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut cur = vec![S::start_non_terminal()];
        // Current length in non-terminals.
        let mut length = 0;

        loop {
            let non_terminal_idx: Vec<_> = (0..cur.len())
                .filter(|i| S::is_non_terminal(cur[*i]))
                .collect();
            if non_terminal_idx.is_empty() {
                assert!(cur.len() == length);
                return Some(cur.into_iter().map(|it| S::from_num(it)).collect());
            }
            let id = non_terminal_idx[self.rng.gen::<usize>() % non_terminal_idx.len()];
            let productions = S::get_productions(cur[id]);
            assert!(!productions.is_empty());

            let prod_id = if length >= self.min_lentgth && length < self.max_length {
                // Choose a random production.
                let production_id = self.rng.gen::<usize>() % productions.len();
                production_id
            } else {
                let (with_non_terminals, without_non_terminals): (Vec<_>, Vec<_>) =
                    (0..productions.len()).partition(|i| {
                        if productions[*i].is_none() {
                            false
                        } else {
                            productions[*i]
                                .as_ref()
                                .unwrap()
                                .iter()
                                .any(|t| S::is_non_terminal(*t))
                        }
                    });
                if length < self.min_lentgth {
                    // Try to thoose a production with non-terminals.
                    take_random_index(&with_non_terminals, &without_non_terminals, &mut self.rng)
                } else {
                    // length >= self.max_length
                    // Try to choose a production without non-terminals.
                    take_random_index(&without_non_terminals, &with_non_terminals, &mut self.rng)
                }
            };

            length += apply_production::<S>(&mut cur, id, &productions[prod_id]) as usize;
        }
    }
}
