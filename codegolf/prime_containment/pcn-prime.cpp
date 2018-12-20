#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <array>

using namespace std;

void debug_dummy(...) {
}

#ifndef INFO
//#  define INFO(...) fprintf(stderr, __VA_ARGS__)
#  define INFO debug_dummy
#endif

#ifndef DEBUG
//#    define DEBUG(...) fprintf(stderr, __VA_ARGS__)
#  define DEBUG debug_dummy
#endif

bool is_prime(size_t n)
{
    for (size_t d = 2; d * d <= n; ++d) {
        if (n % d == 0) {
            return false;
        }
    }
    return true;
}

// bitset, works for up to 64 strings
using bitset_t = uint64_t;
const size_t bitset_bits = 64;

// Find position of n-th set bit of x
uint64_t bit_select(uint64_t x, size_t n) {
#ifdef __BMI2__
    // Bug: GCC doesn't seem to provide the _pdep_u64 intrinsic,
    // despite what its manual claims. Neither does Clang!
    //size_t r = _pdep_u64(ccontext_t(1) << new_context, ccontext1);
    size_t r;
    // NB: actual operand order is %2, %1 despite the intrinsic taking %1, %2
    asm ("pdep %2, %1, %0"
         : "=r" (r)
         : "r" (uint64_t(1) << n), "r" (x)
         );
    return __builtin_ctzll(r);
#else
#  warning "bit_select: no x86 BMI2 instruction set, falling back to slow code"
    size_t k = 0, m = 0;
    for (; m < 64; ++m) {
        if (x & (uint64_t(1) << m)) {
            if (k == n) {
                break;
            }
            ++k;
        }
    }
    return m;
#endif
}

#ifndef likely
#  define likely(x) __builtin_expect(x, 1)
#endif
#ifndef unlikely
#  define unlikely(x) __builtin_expect(x, 0)
#endif

// Return the shortest string that begins with a and ends with b
string join_strings(string a, string b) {
    for (size_t overlap = min(a.size(), b.size()); overlap > 0; --overlap) {
        if (a.substr(a.size() - overlap) == b.substr(0, overlap)) {
            return a + b.substr(overlap);
        }
    }
    return a + b;
}

vector <string> dedup_items(string context0, vector <string> items)
{
    vector <string> items2;
    for (size_t i = 0; i < items.size(); ++i) {
        bool dup = false;
        if (context0.find(items[i]) != string::npos) {
                dup = true;
        } else {
            for (size_t j = 0; j < items.size(); ++j) {
                if (items[i] == items[j]?
                    i > j
                        : items[j].find(items[i]) != string::npos) {
                    dup = true;
                    break;
                }
            }
        }
        if (!dup) {
            items2.push_back(items[i]);
        }
    }
    return items2;
}

// Table entry used in main solver
const size_t solver_max_item_set = bitset_bits - 8;
struct Solver_entry
{
    uint8_t score : 8;
    bitset_t items : solver_max_item_set;
    bitset_t context;

    Solver_entry()
    {
        score = 0xff;
        items = 0;
        context = 0;
    }
    bool is_empty() const {
        return score == 0xff;
    }
};

// Simple hash table to avoid stdlib overhead
struct Solver_table
{
    vector <Solver_entry> t;
    size_t t_bits;
    size_t size_;
    size_t num_probes_;

    Solver_table()
    {
        // 256 slots initially -- this needs to be not too small
        // so that the load factor formula in update_score works
        t_bits = 8;
        size_ = 0;
        num_probes_ = 0;
        resize(t_bits);
    }
    static size_t entry_hash(bitset_t items, bitset_t context)
    {
        uint64_t h = 0x3141592627182818ULL;
        // Add context first, since its bits are generally
        // less well distributed than items
        h += context;
        h ^= h >> 23;
        h *= 0x2127599bf4325c37ULL;
        h ^= h >> 47;
        h += items;
        h ^= h >> 23;
        h *= 0x2127599bf4325c37ULL;
        h ^= h >> 47;
        return h;
    }
    size_t probe_index(size_t hash) const {
        return hash & ((size_t(1) << t_bits) - 1);
    }
    void resize(size_t t2_bits)
    {
        assert (size_ < size_t(1) << t2_bits);
        vector <Solver_entry> t2(size_t(1) << t2_bits);
        for (auto entry: t) {
            if (!entry.is_empty()) {
                size_t h = entry_hash(entry.items, entry.context);
                size_t mask = (size_t(1) << t2_bits) - 1;
                size_t idx = h & mask;
                while (!t2[idx].is_empty()) {
                    idx = (idx + 1) & mask;
                    ++num_probes_;
                }
                t2[idx] = entry;
            }
        }
        t.swap(t2);
        t_bits = t2_bits;
    }
    uint8_t update_score(bitset_t items, bitset_t context, uint8_t score)
    {
        // Ensure we can insert a new item without resizing
        assert (size_ < t.size());

        size_t index = probe_index(entry_hash(items, context));
        size_t mask = (size_t(1) << t_bits) - 1;
        for (size_t p = 0; p < t.size(); ++p, index = (index + 1) & mask) {
            ++num_probes_;
            if (likely(t[index].items == items && t[index].context == context)) {
                t[index].score = max(t[index].score, score);
                return t[index].score;
            }
            if (t[index].is_empty()) {
                // add entry
                t[index].score = score;
                t[index].items = items;
                t[index].context = context;
                ++size_;
                // load factor 4/5 -- ideally 2-3 average probes per lookup
                if (5*size_ > 4*t.size()) {
                    resize(t_bits + 1);
                }
                return score;
            }
        }
        assert (false && "bug: hash table probe loop");
    }
    size_t size() const {
        return size_;
    }
    void swap(Solver_table table)
    {
        t.swap(table.t);
        ::swap(size_, table.size_);
        ::swap(t_bits, table.t_bits);
        ::swap(num_probes_, table.num_probes_);
    }
};

/*
 * Main solver code.
 */
struct Solver
{
    // Inputs
    vector <string> items;
    string context0;
    size_t context0_index;

    // Mapping between strings and indices
    vector <string> context_to_string;
    unordered_map <string, size_t> string_to_context;

    // Items that have context-free prefixes, i.e. prefixes that
    // never overlap with the end of other items nor context0
    vector <bool> contextfree;

    // Precomputed contexts (suffixes) for each item
    vector <size_t> item_context;
    // Precomputed updates: (context, string) to overlap amount
    vector <vector <size_t>> join_overlap;

    Solver(vector <string> items, string context0)
        :items(items), context0(context0)
    {
        items = dedup_items(context0, items);
        init_context_();
    }

    void init_context_()
    {
        /*
         * Generate all relevant item-item contexts.
         *
         * At this point, we know that no item is a substring of
         * another, nor of context0. This means that the only contexts
         * we need to care about, are those generated from maximal join
         * overlaps between any two items.
         *
         * Proof:
         * Suppose that the shortest containing string needs some other
         * kind of context. Maybe it depends on a context spanning
         * three or more items, say X,Y,Z. But if Z ends after Y and
         * interacts with X, then Y must be a substring of Z.
         * This cannot happen, because we removed all substrings.
         *
         * Alternatively, it depends on a non-maximal join overlap
         * between two strings, say X,Y. But if this overlap does not
         * interact with any other string, then we could maximise it
         * and get a shorter solution. If it does, then call this
         * other string Z. We would get the same contradiction as in
         * the previous case with X,Y,Z.
         */
        size_t N = items.size();
        vector <size_t> max_prefix_overlap(N), max_suffix_overlap(N);
        size_t context0_suffix_overlap = 0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                string joined = join_strings(items[j], items[i]);
                size_t overlap = items[j].size() + items[i].size() - joined.size();
                string context = items[i].substr(0, overlap);
                max_prefix_overlap[i] = max(max_prefix_overlap[i], overlap);
                max_suffix_overlap[j] = max(max_suffix_overlap[j], overlap);

                if (string_to_context.find(context) == string_to_context.end()) {
                    string_to_context[context] = context_to_string.size();
                    context_to_string.push_back(context);
                }
            }

            // Context for initial join with context0
            {
                string joined = join_strings(context0, items[i]);
                size_t overlap = context0.size() + items[i].size() - joined.size();
                string context = items[i].substr(0, overlap);
                max_prefix_overlap[i] = max(max_prefix_overlap[i], overlap);
                context0_suffix_overlap = max(context0_suffix_overlap, overlap);

                if (string_to_context.find(context) == string_to_context.end()) {
                    string_to_context[context] = context_to_string.size();
                    context_to_string.push_back(context);
                }
            }
        }
        // Now compute all canonical trailing contexts
        context0_index = string_to_context[
                           context0.substr(context0.size() - context0_suffix_overlap)];
        item_context.resize(N);
        for (size_t i = 0; i < N; ++i) {
            item_context[i] = string_to_context[
                                items[i].substr(items[i].size() - max_suffix_overlap[i])];
        }

        // Now detect context-free items
        contextfree.resize(N);
        for (size_t i = 0; i < N; ++i) {
            contextfree[i] = (max_prefix_overlap[i] == 0);
            if (contextfree[i]) {
                DEBUG("  contextfree: %s\n", items[i].c_str());
            }
        }

        // Now compute all possible overlap amounts
        join_overlap.resize(context_to_string.size(), vector <size_t> (N));
        for (size_t c_index = 0; c_index < context_to_string.size(); ++c_index) {
            const string& context = context_to_string[c_index];
            for (size_t i = 0; i < N; ++i) {
                string joined = join_strings(context, items[i]);
                size_t overlap = context.size() + items[i].size() - joined.size();
                join_overlap[c_index][i] = overlap;
            }
        }
    }

    // Main solver.
    // Returns length of shortest string containing all items starting
    // from context0 (context0's length not included).
    size_t solve() const
    {
        size_t N = items.size();

        // Length, if joined without overlaps. We try to improve this by
        // finding overlaps in the main iteration
        size_t base_length = 0;
        for (auto s: items) {
            base_length += s.size();
        }

        // Now take non-context-free items. We will only need to search
        // over these items.
        vector <size_t> search_items;
        for (size_t i = 0; i < N; ++i) {
            if (!contextfree[i]) {
                search_items.push_back(i);
            }
        }
        size_t N_search = search_items.size();

        /*
         * Some groups of strings have the same context transitions.
         * For example "17", "107", "127", "167" all have an initial
         * context of "1" and a trailing context of "7", no other
         * overlaps are possible with other primes.
         *
         * We group these strings and treat them as indistinguishable
         * during the main algorithm.
         */
        auto eq_context = [&](size_t i, size_t j) {
            if (item_context[i] != item_context[j]) {
                return false;
            }
            for (size_t ci = 0; ci < context_to_string.size(); ++ci) {
                if (join_overlap[ci][i] != join_overlap[ci][j]) {
                    return false;
                }
            }
            return true;
        };
        vector <size_t> eq_context_group(N_search, size_t(-1));
        for (size_t si = 0; si < N_search; ++si) {
            for (size_t sj = si-1; sj+1 > 0; --sj) {
                size_t i = search_items[si], j = search_items[sj];
                if (!contextfree[j] && eq_context(i, j)) {
                    DEBUG("  eq context: %s =c= %s\n", items[i].c_str(), items[j].c_str());
                    eq_context_group[si] = sj;
                    break;
                }
            }
        }

        // Figure out the combined context size. A combined context has
        // one entry for each context-free item plus one for context0.
        size_t ccontext_size = N - N_search + 1;

        // Assert that various parameters all fit into our data types
        using ccontext_t = bitset_t;
        assert (context_to_string.size() + ccontext_size <= bitset_bits);
        assert (N_search <= solver_max_item_set);
        assert (base_length < 0xff);

        // Initial combined context.
        unordered_map <size_t, size_t> cc0_full;
        ++cc0_full[context0_index];
        for (size_t i = 0; i < N; ++i) {
            if (contextfree[i]) {
                ++cc0_full[item_context[i]];
            }
        }
        // Now pack into unary-encoded bitset. The bitset stores the
        // count for each context as <count> number of 0 bits,
        // followed by a 1 bit.
        ccontext_t cc0 = 0;
        for (size_t ci = 0, b = 0; ci < context_to_string.size(); ++ci, ++b) {
            b += cc0_full[ci];
            cc0 |= ccontext_t(1) << b;
        }

        // Map from (item set, context) to maximum achievable overlap
        Solver_table k_solns;
        // Base case: cc0 with empty set
        k_solns.update_score(0, cc0, 0);

        // Now start dynamic programming. k is current subset size
        size_t eq_context_groups = 0;
        for (size_t g: eq_context_group) eq_context_groups += (g != size_t(-1));
        if (context0.empty()) {
            INFO("solve: N=%zu, N_search=%zu, ccontext_size=%zu, #contexts=%zu, #eq_context_groups=%zu\n",
                 N, N_search, ccontext_size, context_to_string.size(), eq_context_groups);
        } else {
            DEBUG("solve: context=%s, N=%zu, N_search=%zu, ccontext_size=%zu, #contexts=%zu, #eq_context_groups=%zu\n",
                  context0.c_str(), N, N_search, ccontext_size, context_to_string.size(), eq_context_groups);
        }
        for (size_t k = 0; k < N_search; ++k) {
            decltype(k_solns) k1_solns;

            // The main bottleneck of this program is updating k1_solns,
            // which (for larger N) becomes a huge table.
            // We use a prefetch queue to reduce memory latency.
            const size_t prefetch = 8;
            array <Solver_entry, prefetch> entry_queue;
            size_t update_i = 0;

            // Iterate every k-subset
            for (Solver_entry entry: k_solns.t) {
                if (entry.is_empty()) continue;

                bitset_t s = entry.items;
                ccontext_t ccontext = entry.context;
                size_t overlap = entry.score;

                // Try adding a new item
                for (size_t si = 0; si < N_search; ++si) {
                    bitset_t s1 = s | bitset_t(1) << si;
                    if (s == s1) {
                        continue;
                    }
                    // Add items in each eq_context_group sequentially
                    if (eq_context_group[si] != size_t(-1) &&
                        !(s & bitset_t(1) << eq_context_group[si])) {
                        continue;
                    }
                    size_t i = search_items[si]; // actual item index

                    size_t new_context = item_context[i];
                    // Increment ccontext's count for new_context.
                    // We need to find its delimiter 1 bit
                    size_t bit_n = bit_select(ccontext, new_context);
                    ccontext_t ccontext_n =
                        ((ccontext & ((ccontext_t(1) << bit_n) - 1))
                         | ((ccontext >> bit_n << (bit_n + 1))));

                    // Select non-empty sub-contexts to substitute for new_context
                    for (size_t ci = 0, bit1 = 0, count;
                         ci < context_to_string.size();
                         ++ci, bit1 += count + 1)
                    {
                        assert (ccontext_n >> bit1);
                        count = __builtin_ctzll(ccontext_n >> bit1);
                        if (!count
                            // We just added new_context; we can only remove an existing
                            // context entry there i.e. there must be at least two now
                            || (ci == new_context && count < 2)) {
                            continue;
                        }

                        // Decrement ci in ccontext_n
                        bitset_t ccontext1 =
                            ((ccontext_n & ((ccontext_t(1) << bit1) - 1))
                             | ((ccontext_n >> (bit1 + 1)) << bit1));

                        size_t overlap1 = overlap + join_overlap[ci][i];

                        // do previous prefetched update
                        if (update_i >= prefetch) {
                            Solver_entry entry = entry_queue[update_i % prefetch];
                            k1_solns.update_score(entry.items, entry.context, entry.score);
                        }

                        // queue the current update and prefetch
                        Solver_entry entry1;
                        size_t probe_index = k1_solns.probe_index(Solver_table::entry_hash(s1, ccontext1));
                        __builtin_prefetch(&k1_solns.t[probe_index]);
                        entry1.items = s1;
                        entry1.context = ccontext1;
                        entry1.score = overlap1;
                        entry_queue[update_i % prefetch] = entry1;

                        ++update_i;
                    }
                }
            }

            // do remaining queued updates
            for (size_t j = 0; j < min(update_i, prefetch); ++j) {
                Solver_entry entry = entry_queue[j];
                k1_solns.update_score(entry.items, entry.context, entry.score);
            }

            if (context0.empty()) {
                INFO("  hash stats: |solns[%zu]| = %zu, %zu lookups, %zu probes\n",
                     k+1, k1_solns.size(), update_i, k1_solns.num_probes_);
            } else {
                DEBUG("  hash stats: |solns[%zu]| = %zu, %zu lookups, %zu probes\n",
                      k+1, k1_solns.size(), update_i, k1_solns.num_probes_);
            }
            k_solns.swap(k1_solns);
        }

        // Overall solution
        size_t max_overlap = 0;
        for (Solver_entry entry: k_solns.t) {
            if (entry.is_empty()) continue;
            max_overlap = max(max_overlap, size_t(entry.score));
        }
        return base_length - max_overlap;
    }
};

// Wrapper for Solver that also finds the smallest solution string
string smallest_containing_string(vector <string> items)
{
    items = dedup_items("", items);

    size_t soln_length;
    {
        Solver solver(items, "");
        soln_length = solver.solve();
    }
    DEBUG("Found solution length: %zu\n", soln_length);

    string soln;
    vector <string> remaining_items = items;
    while (remaining_items.size() > 1) {
        // Add all possible next items, in lexicographic order
        vector <pair <string, size_t>> next_solns;
        for (size_t i = 0; i < remaining_items.size(); ++i) {
            const string& item = remaining_items[i];
            next_solns.push_back(make_pair(join_strings(soln, item), i));
        }
        assert (next_solns.size() == remaining_items.size());
        sort(next_solns.begin(), next_solns.end());

        // Now try every item in order
        bool found_next = false;
        for (auto ns: next_solns) {
            size_t i;
            string next_soln;
            tie(next_soln, i) = ns;
            DEBUG("Trying: %s + %s -> %s\n",
                  soln.c_str(), remaining_items[i].c_str(), next_soln.c_str());
            vector <string> next_remaining;
            for (size_t j = 0; j < remaining_items.size(); ++j) {
                if (next_soln.find(remaining_items[j]) == string::npos) {
                    next_remaining.push_back(remaining_items[j]);
                }
            }

            Solver solver(next_remaining, next_soln);
            size_t next_size = solver.solve();
            DEBUG("  ... next_size: %zu + %zu =?= %zu\n", next_size, next_soln.size(), soln_length);
            if (next_size + next_soln.size() == soln_length) {
                INFO("  found next item: %s\n", remaining_items[i].c_str());
                soln = next_soln;
                remaining_items = next_remaining;
                // found lexicographically smallest solution, break now
                found_next = true;
                break;
            }
        }
        assert (found_next);
    }
    soln = join_strings(soln, remaining_items[0]);

    return soln;
}

int main()
{
    string prev_soln;
    vector <string> items;
    size_t p = 1;
    for (size_t N = 1;; ++N) {
        for (++p; items.size() < N; ++p) {
            if (is_prime(p)) {
                char buf[99];
                snprintf(buf, sizeof buf, "%zu", p);
                items.push_back(buf);
                break;
            }
        }

        // Try to reuse previous solution (this works for N=11,30,32...)
        string soln;
        if (prev_soln.find(items.back()) != string::npos) {
            soln = prev_soln;
        } else {
            soln = smallest_containing_string(items);
        }
        printf("%lu %s\n", N, soln.c_str());
        prev_soln = soln;
    }
}