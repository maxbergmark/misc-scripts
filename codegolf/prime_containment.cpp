#include <cassert>
#include <cstdlib>
#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <algorithm>

using namespace std;

void debug_dummy(...) {
}

//#define DEBUG(...) fprintf(stderr, __VA_ARGS__)
#define DEBUG debug_dummy

bool is_prime(size_t n)
{
    if (n < 2) return false;
    for (size_t d = 2; d * d <= n; ++d) {
        if (n % d == 0) return false;
    }
    return true;
}

// bitset, works for up to 64 strings
using str_set_t = uint64_t;
const unsigned str_set_bits = 64;

inline size_t popcount(size_t x) {
    return x? __builtin_popcount(x) : 0;
}

// from Bit Twiddling Hacks collection
inline str_set_t next_bit_permutation(str_set_t v) {
    str_set_t t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change,
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
}

// compact index into a string table
// size_t is overkill, but the size isn't important
using context_t = size_t;

inline size_t str_set_size(str_set_t x) {
    return popcount(x);
}
inline str_set_t str_set_empty() {
    return 0;
}
inline str_set_t str_set_insert(str_set_t x, size_t i) {
    assert (i < str_set_bits);
    return x | (str_set_t(1) << i);
}
inline str_set_t str_set_erase(str_set_t x, size_t i) {
    assert (i < str_set_bits);
    return x & ~(str_set_t(1) << i);
}
inline str_set_t str_set_single(size_t i) {
    return str_set_insert(str_set_empty(), i);
}
inline bool str_set_member(str_set_t x, size_t i) {
    assert (i < str_set_bits);
    return x & (str_set_t(1) << i);
}

size_t comb(size_t n, size_t k) {
    size_t x = 1;
    for (size_t i = 1; i <= k; ++i) {
        x *= n - i + 1;
        x /= i;
    }
    return x;
}

// Return the shortest string that starts with a and ends with b
string join_strings(string a, string b) {
    for (size_t offset = min(a.size(), b.size()); offset > 0; --offset) {
        if (a.substr(a.size() - offset) == b.substr(0, offset)) {
            return a + b.substr(offset);
        }
    }
    // no overlap
    return a + b;
}

struct subset_index
{
    size_t n, k;
    size_t rank_split;

    vector <size_t> rank_large;
    vector <vector <size_t>> rank_small;

    subset_index(size_t n, size_t k):
        n(n), k(k), rank_split(n/2)
    {
        vector <size_t> small_size(k+1, 0);
        rank_small.resize(k+1, vector <size_t> (size_t(1) << rank_split));
        for (str_set_t x = 0; x < str_set_t(1) << rank_split; ++x) {
            size_t sz = popcount(x);
            if (sz <= k) {
                rank_small[sz][x] = small_size[sz]++;
            }
        }

        size_t large_offset = 0;
        rank_large.resize(size_t(1) << (n - rank_split));
        for (str_set_t x = 0; x < str_set_t(1) << (n - rank_split); ++x) {
            size_t sz = popcount(x);
            if (sz <= k) {
                rank_large[x] = large_offset;
                large_offset += small_size[k - sz];
            }
        }
        assert (large_offset == comb(n, k));
    }

    size_t rank(str_set_t x) const {
        size_t large = x >> rank_split, small = x & ((str_set_t(1) << rank_split) - 1);
        size_t r = rank_large.at(large) + rank_small.at(popcount(small))[small];
        return r;
    }

    void swap(subset_index& si) {
        ::swap(n, si.n);
        ::swap(k, si.k);
        ::swap(rank_split, si.rank_split);
        rank_large.swap(si.rank_large);
        rank_small.swap(si.rank_small);
    }
};

struct Solver
{
    vector <string> items;
    set <char> alphabet;
    size_t context_length;
    vector <string> contexts; // context_t -> string
    unordered_map <string, context_t> context_index; // string -> context_t
    // Solution must start with this context. Used for self reduction
    string context0;
    // Precomputed context updates
    vector <vector <pair <context_t, size_t>>> context_update;

    Solver(vector <string> items, string context0)
        : items(items), context0(context0)
    {
        init_context();
    }

    void generate_contexts_()
    {
        vector <string> cs;
        cs.push_back("");
        contexts.push_back(cs[0]);
        context_index[""] = 0;
        for (size_t l = 1; l <= context_length; ++l) {
            vector <string> cs2;
            for (string c: cs) {
                for (char a: alphabet) {
                    c.push_back(a);
                    cs2.push_back(c);
                    c.pop_back();
                }
            }
            for (const string c: cs2) {
                context_index[c] = contexts.size();
                contexts.push_back(c);
            }
            cs.swap(cs2);
        }
        if (context_index.find(context0) == context_index.end()) {
            context_index[context0] = contexts.size();
            contexts.push_back(context0);
        }
    }

    void generate_context_update_()
    {
        for (size_t i = 0; i < items.size(); ++i) {
            const string& item = items[i];
            vector <pair <context_t, size_t>> upd;
            for (size_t c = 0; c < contexts.size(); ++c) {
                const string& context = contexts[c];
                string new_context = join_strings(context, item);
                size_t added_length = new_context.size() - context.size();
                if (new_context.size() > context_length) {
                    new_context = new_context.substr(new_context.size() - context_length);
                }
                upd.push_back(make_pair(context_index[new_context], added_length));
            }
            context_update.push_back(upd);
        }
    }

    void init_context()
    {
        alphabet.clear();
        contexts.clear();
        context_index.clear();
        context_update.clear();

        context_length = 1;
        for (const auto& item: items) {
            for (const auto& item2: items) {
                if (item == item2) continue;
                size_t overlap = item.size() + item2.size() - join_strings(item, item2).size();
                context_length = max(context_length, overlap + 1);
            }
        }
        DEBUG("context_length = %zu\n", context_length);

        // Get alphabet for context generation
        assert (!items.empty());
        for (const auto& item: items) {
            assert (!item.empty());
            for (char c: item) {
                alphabet.insert(c);
            }
        }

        // We only care about the last few context characters
        if (context0.size() > context_length) {
            context0 = context0.substr(context0.size() - context_length);
        }
        generate_contexts_();
        generate_context_update_();
    }

    size_t solve()
    {
        using length_t = uint8_t;

        // contiguous cache
        pair <size_t, size_t> context_update[items.size()][contexts.size()];
        for (size_t i = 0; i < items.size(); ++i) {
            for (size_t c = 0; c < contexts.size(); ++c) {
                context_update[i][c] = this->context_update[i][c];
            }
        }

        vector <vector <length_t>> k_solns;
        subset_index k_index(items.size(), 0);
        for (size_t k = 0; k < items.size(); ++k) {
            size_t comb_k1 = comb(items.size(), k+1);
            vector <vector <length_t>> k1_solns(contexts.size());
            subset_index k1_index(items.size(), k+1);

            if (k == 0) {
                // Table is empty. Fill it with singleton subsets
                // based on the initial context0
                context_t c0 = context_index[context0];
                for (size_t i = 0; i < items.size(); ++i) {
                    str_set_t s = str_set_single(i);
                    context_t c1;
                    size_t length;
                    tie(c1, length) = context_update[i][c0];
                    assert (length_t(length) == length);
                    // lazy init
                    if (k1_solns[c1].empty()) {
                        k1_solns[c1].resize(comb_k1);
                    }
                    k1_solns[c1][k1_index.rank(s)] = length;
                }
            } else {
                // For each subset and context result in k_solns,
                // add one item to the subset and update k1_solns.
                for (context_t c = 0; c < contexts.size(); ++c) {
                    // Here we rely on the fact that subset_index
                    // enumerates subsets in lexicographic order,
                    // so s will have the correct subset for each iter
                    str_set_t s = (str_set_t(1) << k) - 1;
                    for (size_t length: k_solns[c]) {
                        // ensure we have a real entry
                        if (length) {
                            for (size_t i = 0; i < items.size(); ++i) {
                                if (str_set_member(s, i)) {
                                    continue;
                                }
                                // New item i
                                str_set_t s1 = str_set_insert(s, i);
                                context_t c1;
                                size_t length_diff;
                                tie(c1, length_diff) = context_update[i][c];
                                size_t length1 = length + length_diff;
                                assert (length_t(length1) == length1);

                                // lazy init
                                if (k1_solns[c1].empty()) {
                                    k1_solns[c1].resize(comb_k1);
                                }
                                // Update best length
                                length_t& soln1 = k1_solns[c1][k1_index.rank(s1)];
                                if (soln1 == 0 || soln1 > length1) {
                                    soln1 = length1;
                                }
                            }
                        }

                        s = next_bit_permutation(s);
                    }
                }
            }

            k_solns.swap(k1_solns);
            k_index.swap(k1_index);
        }

        // Return the minimal string length
        size_t best_length = 0;
        for (context_t c = 0; c < contexts.size(); ++c) {
            if (k_solns.at(c).empty()) continue;
            length_t soln = k_solns[c][0];
            assert (soln > 0);
            if (!best_length || best_length > soln) {
                best_length = soln;
            }
        }
        return best_length;
    }
};

string prime_container(vector <string> items)
{
    // Remove items which are substrings of other items
    vector <string> dedup_items;
    for (size_t i = 0; i < items.size(); ++i) {
        bool dupe = false;
        for (size_t j = 0; j < items.size(); ++j) {
            if (i != j && items[j].find(items[i]) != string::npos) {
                dupe = true;
            }
        }
        if (dupe) {
            DEBUG("removing redundant item: %s\n", items[i].c_str());
        } else {
            dedup_items.push_back(items[i]);
        }
    }
    items = dedup_items;

    size_t soln_length;
    {
        Solver solver(items, "");
        soln_length = solver.solve();
    }
    DEBUG("Found solution length: %zu\n", soln_length);

    string soln;
    vector <string> remaining_items = items;
    for (size_t i = 0; i < items.size() - 1; ++i) {
        // Add all possible next items, in lexicographic order
        vector <pair <string, size_t>> next_solns;
        for (size_t j = 0; j < remaining_items.size(); ++j) {
            const string& item = remaining_items[j];
            string next_soln;
            // Add new item optimally -- possibly in the middle (no-op)
            if (soln.find(item) != string::npos) {
                // We can actually skip to the next i here, but this won't
                // happen early enough in our iteration to be worth it
                next_soln = soln;
                break;
            } else {
                next_soln = join_strings(soln, item);
            }
            next_solns.push_back(make_pair(next_soln, j));
        }
        assert (next_solns.size() == remaining_items.size());
        sort(next_solns.begin(), next_solns.end());

        // Now process the items in order
        bool found_next = false;
        for (auto ns: next_solns) {
            size_t j;
            string next_soln;
            tie(next_soln, j) = ns;
            DEBUG("Trying: %s + %s -> %s\n",
                  soln.c_str(), remaining_items[j].c_str(), next_soln.c_str());
            vector <string> next_remaining;
            next_remaining.insert(next_remaining.begin(),
                                  remaining_items.begin(), remaining_items.begin() + j);
            next_remaining.insert(next_remaining.begin(),
                                  remaining_items.begin() + j + 1, remaining_items.end());

            Solver solver(next_remaining, next_soln);
            size_t next_size = solver.solve();
            DEBUG("  ... next size = %zu\n", next_size);
            if (next_size + next_soln.size() == soln_length) {
                DEBUG("  (success)\n");
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
    for (size_t N = 1; N <= 40; ++N) {
        for (++p; items.size() < N; ++p) {
            if (is_prime(p)) {
                char buf[99];
                snprintf(buf, sizeof buf, "%zu", p);
                items.push_back(buf);
                break;
            }
        }

        // Try to reuse previous solution (this works for N=11,30,...)
        string soln;
        if (prev_soln.find(items.back()) != string::npos) {
            soln = prev_soln;
        } else {
            soln = prime_container(items);
        }
        printf("%2zu, %s\n", N, soln.c_str());
        prev_soln = soln;
    }
}