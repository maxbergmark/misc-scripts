import multiprocessing
from ortools.sat.python import cp_model
import time

def superstring(strings):
    def gen_prefixes(s):
        for i in range(len(s)):
            a = s[:i]
            if a in affixes:
                yield a

    def gen_suffixes(s):
        for i in range(1, len(s) + 1):
            a = s[i:]
            if a in affixes:
                yield a

    def solve():
        def find_string(s):
            found_strings.add(s)
            for i in range(1, len(s) + 1):
                a = s[i:]
                if (
                    a in affixes
                    and a not in found_affixes
                    and solver.Value(suffix[s, a])
                ):
                    found_affixes.add(a)
                    q.append(a)
                    break

        def cut(skip):
            model.AddBoolOr(
                skip
                + [
                    suffix[s, a]
                    for s in found_strings
                    for a in gen_suffixes(s)
                    if a not in found_affixes
                ]
                + [
                    prefix[a, s]
                    for s in unused_strings
                    if s not in found_strings
                    for a in gen_prefixes(s)
                    if a in found_affixes
                ]
            )
            model.AddBoolOr(
                skip
                + [
                    suffix[s, a]
                    for s in unused_strings
                    if s not in found_strings
                    for a in gen_suffixes(s)
                    if a in found_affixes
                ]
                + [
                    prefix[a, s]
                    for s in found_strings
                    for a in gen_prefixes(s)
                    if a not in found_affixes
                ]
            )

        def search():
            while q:
                a = q.pop()
                for s in prefixed[a]:
                    if (
                        s in unused_strings
                        and s not in found_strings
                        and solver.Value(prefix[a, s])
                    ):
                        find_string(s)
            return not (unused_strings - found_strings)

        while True:
            if solver.Solve(model) != cp_model.OPTIMAL:
                raise RuntimeError("Solve failed")

            found_strings = set()
            found_affixes = set()
            if part is None:
                found_affixes.add("")
                q = [""]
            else:
                part_ix = solver.Value(part)
                p, next_affix, next_string = parts[part_ix]
                q = []
                find_string(next_string)
            if search():
                break

            if part is not None:
                if part_ix not in partb:
                    partb[part_ix] = model.NewBoolVar("partb%s_%s" % (step, part_ix))
                    model.Add(part == part_ix).OnlyEnforceIf(partb[part_ix])
                    model.Add(part != part_ix).OnlyEnforceIf(partb[part_ix].Not())
                cut([partb[part_ix].Not()])
                if last_string is None:
                    found_affixes.add(next_affix)
                else:
                    find_string(last_string)
                q.append(next_affix)
                if search():
                    continue

            cut([])

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 4
    affixes = {s[:i] for s in strings for i in range(len(s))} & {
        s[i:] for s in strings for i in range(1, len(s) + 1)
    }
    prefixed = {}
    for s in strings:
        for a in gen_prefixes(s):
            prefixed.setdefault(a, []).append(s)
    suffixed = {}
    for s in strings:
        for a in gen_suffixes(s):
            suffixed.setdefault(a, []).append(s)
    unused_strings = set(strings)
    last_string = None
    part = None

    model = cp_model.CpModel()
    prefix = {
        (a, s): model.NewBoolVar("prefix_%s_%s" % (a, s))
        for a in affixes
        for s in prefixed[a]
    }
    suffix = {
        (s, a): model.NewBoolVar("suffix_%s_%s" % (s, a))
        for a in affixes
        for s in suffixed[a]
    }
    for s in strings:
        model.Add(sum(prefix[a, s] for a in gen_prefixes(s)) == 1)
        model.Add(sum(suffix[s, a] for a in gen_suffixes(s)) == 1)
    for a in affixes:
        model.Add(
            sum(suffix[s, a] for s in suffixed[a])
            == sum(prefix[a, s] for s in prefixed[a])
        )

    length = sum(prefix[a, s] * (len(s) - len(a)) for a in affixes for s in prefixed[a])
    model.Minimize(length)
    solve()
    model.Add(length == solver.Value(length))

    out = ""
    for step in range(len(strings)):
        in_parts = set()
        parts = []
        for a in [""] if last_string is None else gen_suffixes(last_string):
            for s in prefixed[a]:
                if s in unused_strings and s not in in_parts:
                    in_parts.add(s)
                    parts.append((s[len(a) :], a, s))
        parts.sort()
        part = model.NewIntVar(0, len(parts) - 1, "part%s" % step)
        partb = {}
        for part_ix, (p, a, s) in enumerate(parts):
            if last_string is not None:
                model.Add(part != part_ix).OnlyEnforceIf(suffix[last_string, a].Not())
            model.Add(part != part_ix).OnlyEnforceIf(prefix[a, s].Not())
        model.Minimize(part)
        solve()
        part_ix = solver.Value(part)
        model.Add(part == part_ix)
        p, a, last_string = parts[part_ix]
        unused_strings.remove(last_string)
        out += p
    return out


def gen_primes():
    yield 2
    n = 3
    d = {}
    for p in gen_primes():
        p2 = p * p
        d[p2] = 2 * p
        while n <= p2:
            if n in d:
                q = d.pop(n)
                m = n + q
                while m in d:
                    m += q
                d[m] = q
            else:
                yield n
            n += 2


def gen_inputs():
    num_primes = 0
    strings = []

    for new_prime in gen_primes():
        num_primes += 1
        new_string = str(new_prime)
        strings = [s for s in strings if s not in new_string] + [new_string]
        yield strings

t0 = time.time()
with multiprocessing.Pool() as pool:
    for i, out in enumerate(pool.imap(superstring, gen_inputs())):
        t1 = time.time()
        print(i + 1, out, t1-t0, flush=True)