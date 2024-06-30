pub enum Either<L, R> {
    Left(L),
    Right(R),
}

// TODO: use an existing crate?
impl<L, R> Either<L, R> {
    pub fn is_left(&self) -> bool {
        match *self {
            Self::Left(_) => true,
            Self::Right(_) => false,
        }
    }

    pub fn is_right(&self) -> bool {
        !self.is_left()
    }

    pub fn left(self) -> Option<L> {
        match self {
            Self::Left(l) => Some(l),
            Self::Right(_) => None,
        }
    }

    pub fn right(self) -> Option<R> {
        match self {
            Self::Left(_) => None,
            Self::Right(r) => Some(r),
        }
    }

    pub fn as_ref(&self) -> Either<&L, &R> {
        match *self {
            Self::Left(ref inner) => Either::Left(inner),
            Self::Right(ref inner) => Either::Right(inner),
        }
    }
}
