module main
import StdEnv,StdOverloadedList,_SystemEnumStrict
import Data.List,Data.Func,Data.Maybe
import Text,Text.GenJSON

// adapted from Data.Set to work with a single specific type, and persist uniqueness
:: Set a = Tip | Bin !Int a !.(Set a) !.(Set a)
derive JSONEncode Set

delta :== 4
ratio :== 2

:: NumberType :== String

makeSetType e = e

:: SetType :== NumberType

toNumberType = toString

//uSingleton :: SetType -> Set
uSingleton x :== (Bin 1 x Tip Tip)

// adapted from Data.Set to work with a single specific type, and persist uniqueness
uFindMin :: !.(Set .a) -> .a
uFindMin (Bin _ x Tip _) = x
uFindMin (Bin _ _ l _)   = uFindMin l

uSize set :== case set of
	Tip = (0, Tip)
	s=:(Bin sz _ _ _) = (sz, s)
	
uMemberSpec :: NumberType !u:(Set NumberType) -> .(.Bool, v:(Set NumberType)), [u <= v]
uMemberSpec x Tip = (False, Tip)
uMemberSpec x set=:(Bin s y l r)
	| sx < sy || sx == sy && x < y
		# (t, l) = uMemberSpec x l
		= (t, Bin s y l r)
		//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceL y l r)
	| sx > sy || sx == sy && x > y
		# (t, r) = uMemberSpec x r
		= (t, Bin s y l r)
		//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceR y l r)
	| otherwise = (True, set)
where
	sx => size x
	sy => size y

uInsertM :: !(a a -> .Bool) -> (a u:(Set a) -> v:(.Bool, w:(Set a))), [v u <= w]
uInsertM cmp = uInsertM`
where
	//uInsertM` :: a (Set a) -> (Bool, Set a)
	uInsertM` x Tip = (False, uSingleton x)
	uInsertM` x set=:(Bin _ y l r)
		| cmp x y//sx < sy || sx == sy && x < y
			# (t, l) = uInsertM` x l
			= (t, uBalanceL y l r)
			//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceL y l r)
		| cmp y x//sx > sy || sx == sy && x > y
			# (t, r) = uInsertM` x r
			= (t, uBalanceR y l r)
			//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceR y l r)
		| otherwise = (True, set)
		
uInsertMCmp :: a !u:(Set a) -> .(.Bool, v:(Set a)) | Enum a, [u <= v]
uInsertMCmp x Tip = (False, uSingleton x)
uInsertMCmp x set=:(Bin _ y l r)
	| x < y
		# (t, l) = uInsertMCmp x l
		= (t, uBalanceL y l r)
		//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceL y l r)
	| x > y
		# (t, r) = uInsertMCmp x r
		= (t, uBalanceR y l r)
		//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceR y l r)
	| otherwise = (True, set)

uInsertMSpec :: NumberType !u:(Set NumberType) -> .(.Bool, v:(Set NumberType)), [u <= v]
uInsertMSpec x Tip = (False, uSingleton x)
uInsertMSpec x set=:(Bin _ y l r)
	| sx < sy || sx == sy && x < y
		# (t, l) = uInsertMSpec x l
		= (t, uBalanceL y l r)
		//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceL y l r)
	| sx > sy || sx == sy && x > y
		# (t, r) = uInsertMSpec x r
		= (t, uBalanceR y l r)
		//= (t, if(t)(\y` l` r` = Bin sz y` l` r`) uBalanceR y l r)
	| otherwise = (True, set)
where
	sx => size x
	sy => size y

// adapted from Data.Set to work with a single specific type, and persist uniqueness
uBalanceL :: .a !u:(Set .a) !v:(Set .a) -> w:(Set .a), [v u <= w]
//a .(Set a) .(Set a) -> .(Set a)
uBalanceL x Tip Tip
	= Bin 1 x Tip Tip
uBalanceL x l=:(Bin _ _ Tip Tip) Tip
	= Bin 2 x l Tip
uBalanceL x l=:(Bin _ lx Tip (Bin _ lrx _ _)) Tip
	= Bin 3 lrx (Bin 1 lx Tip Tip) (Bin 1 x Tip Tip)
uBalanceL x l=:(Bin _ lx ll=:(Bin _ _ _ _) Tip) Tip
	= Bin 3 lx ll (Bin 1 x Tip Tip)
uBalanceL x l=:(Bin ls lx ll=:(Bin lls _ _ _) lr=:(Bin lrs lrx lrl lrr)) Tip
	| lrs < ratio*lls
		= Bin (1+ls) lx ll (Bin (1+lrs) x lr Tip)
	# (lrls, lrl) = uSize lrl
	# (lrrs, lrr) = uSize lrr
	| otherwise
		= Bin (1+ls) lrx (Bin (1+lls+lrls) lx ll lrl) (Bin (1+lrrs) x lrr Tip)
uBalanceL x Tip r=:(Bin rs _ _ _)
	= Bin (1+rs) x Tip r
uBalanceL x l=:(Bin ls lx ll lr) r=:(Bin rs _ _ _)
	| ls > delta*rs
		= uBalanceL` ll lr
	| otherwise
		= Bin (1+ls+rs) x l r
where
	uBalanceL` ll=:(Bin lls _ _ _) lr=:(Bin lrs lrx lrl lrr)
		| lrs < ratio*lls
			= Bin (1+ls+rs) lx ll (Bin (1+rs+lrs) x lr r)
		# (lrls, lrl) = uSize lrl
		# (lrrs, lrr) = uSize lrr
		| otherwise
			= Bin (1+ls+rs) lrx (Bin (1+lls+lrls) lx ll lrl) (Bin (1+rs+lrrs) x lrr r)

// adapted from Data.Set to work with a single specific type, and persist uniqueness
uBalanceR :: .a !u:(Set .a) !v:(Set .a) -> w:(Set .a), [v u <= w]
uBalanceR x Tip Tip
	= Bin 1 x Tip Tip
uBalanceR x Tip r=:(Bin _ _ Tip Tip)
	= Bin 2 x Tip r
uBalanceR x Tip r=:(Bin _ rx Tip rr=:(Bin _ _ _ _))
	= Bin 3 rx (Bin 1 x Tip Tip) rr
uBalanceR x Tip r=:(Bin _ rx (Bin _ rlx _ _) Tip)
	= Bin 3 rlx (Bin 1 x Tip Tip) (Bin 1 rx Tip Tip)
uBalanceR x Tip r=:(Bin rs rx rl=:(Bin rls rlx rll rlr) rr=:(Bin rrs _ _ _))
	| rls < ratio*rrs
		= Bin (1+rs) rx (Bin (1+rls) x Tip rl) rr
	# (rlls, rll) = uSize rll
	# (rlrs, rlr) = uSize rlr
	| otherwise
		= Bin (1+rs) rlx (Bin (1+rlls) x Tip rll) (Bin (1+rrs+rlrs) rx rlr rr)
uBalanceR x l=:(Bin ls _ _ _) Tip
	= Bin (1+ls) x l Tip
uBalanceR x l=:(Bin ls _ _ _) r=:(Bin rs rx rl rr)
	| rs > delta*ls
		= uBalanceR` rl rr
	| otherwise
		= Bin (1+ls+rs) x l r
where
	uBalanceR` rl=:(Bin rls rlx rll rlr) rr=:(Bin rrs _ _ _)
		| rls < ratio*rrs
			= Bin (1+ls+rs) rx (Bin (1+ls+rls) x l rl) rr
		# (rlls, rll) = uSize rll
		# (rlrs, rlr) = uSize rlr
		| otherwise
			= Bin (1+ls+rs) rlx (Bin (1+ls+rlls) x l rll) (Bin (1+rrs+rlrs) rx rlr rr)
			
primes :: [Int]
primes =: [2: [i \\ i <- [3, 5..] | let
		checks :: [Int]
		checks = TakeWhile (\n = i >= n*n) primes
	in All (\n = i rem n <> 0) checks]]

primePrefixes :: [[NumberType]]
primePrefixes =: Tl (Scan removeOverlap [|] [toNumberType p \\ p <- primes])

removeOverlap :: !u:[NumberType] NumberType -> v:[NumberType], [u <= v]
removeOverlap [|] nsub = [|nsub]
removeOverlap [|h: t] nsub
	| indexOf h nsub <> -1
		= removeOverlap t nsub
	| nsub > h
		= [|h: removeOverlap t nsub]
	| otherwise
		= [|nsub, h: Filter (\s = indexOf s nsub == -1) t]

getMergeCandidate :: !NumberType !NumberType -> .Maybe .NumberType
getMergeCandidate a b
	| a == b = Nothing
	| otherwise = first_prefix (max (size a - size b) 0)
where
	sa => size a - 1
	max_len => min sa (size b - 1)
	first_prefix :: !Int -> .Maybe .NumberType
	first_prefix n
		| n > max_len 
			= Nothing
		| b%(0,sa-n) == a%(n,sa)
			= Just (a%(0,n-1) +++. b)
		| otherwise
			= first_prefix (inc n)

mergeString :: !NumberType !NumberType -> .NumberType
mergeString a b = first_prefix (max (size a - size b) 0)
where
	sa => size a - 1
	first_prefix :: !Int -> .NumberType
	first_prefix n
		| b%(0,sa-n) == a%(n,sa)
			= a%(0,n-1) +++. b
		| otherwise
			= first_prefix (inc n)

			
//uFilterSt :: (a -> .b -> .(Bool,.b)) -> .[a] -> .b -> .(.[a],.b)
uFilterSt fn :== uFilterSt`
where
	uFilterSt` :: .[a] .b -> .(.[a], .b)
	uFilterSt` [|] s = ([|], s)
	uFilterSt` [|h:t] s
		# (iff, s) = fn h s
		| not iff
			= uFilterSt` t s
		# (t, s) = uFilterSt` t s
		| otherwise
			= ([|h:t], s)
	
	
:: CombType :== NumberType
// todo: keep track of merges that we make independent of the resulting whole number
//mapCandidatePermsSt :: !.[.[.NumberType]] u:(Set .NumberType) -> v:(Set NumberType), [u <= v]
mapCandidatePermsSt :: ![[NumberType]] (Set NumberType) -> (Set NumberType)
mapCandidatePermsSt [| ] returnSet = returnSet
mapCandidatePermsSt [h:t] returnSet
	#! (mem, returnSet) = uInsertMSpec h` returnSet
	| mem
		= mapCandidatePermsSt t returnSet
	| otherwise
		= mapCandidatePermsSt (merges ++| t) returnSet
where
	//maxlen` = size h`
	h` => foldl mergeString "" h
	merges = [removeOverlap h y \\ x <- h, (Just y) <- Map (getMergeCandidate x) h]
	dropper e s
		# (mem, s) = uInsertM (<) e s
		= (not mem, s)
	

		
containmentNumbersSt = [ uFindMin (mapCandidatePermsSt [|p] Tip)   \\ p <- primePrefixes] 

minFinder :== (\a b = let sa = size a; sb = size b in if(sa == sb) (a < b) (sa < sb))

tests = [removeOverlap h y \\ x <- h, (Just y) <- Map (getMergeCandidate x) h]
where h = primePrefixes !! 100

Start = [(i, ' ', n, "\n") \\ i <- [1..] & n <- containmentNumbersSt]