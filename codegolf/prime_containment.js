'use strict';
const { PerformanceObserver, performance } = require('perf_hooks');

let f = n => {
  let visited = {},
      a, d, k, best, search;

  // build the list of primes, as strings
  for(a = [ '2' ], n--, k = 3; n; k++) {
    for(d = k; k % (d -= 2);) {}
    d == 1 && n-- && a.push(k + '');
  }

  best = a.join('');

  // recursive search function
  (search = (a, n = 0, r = []) => {
    let x, y, i, j, k, s;

    // remove all entries in r[] that can be found in another entry
    r = r.filter((p, i) => !r.some((q, j) => i != j && ~q.indexOf(p)));

    // abort early if this node was already visited
    if(visited[r]) {
      return;
    }

    // otherwise, mark it as visited
    visited[r] = 1;

    // walk through all distinct pairs (x, y) in r[]
    for(i = 0; i < r.length; i++) {
      for(j = i + 1; j < r.length; j++) {
        x = r[i];
        y = r[j];

        // try to merge x and y if:
        // 1) the first k digits of x equal the last k digits of y
        for(k = 1; x.slice(0, k) == y.slice(-k); k++) {
          r[i] = y + x.slice(k);
          search(a, n, r);
        }

        // or:
        // 2) the first k digits of y equal the last k digits of x
        for(k = 1; y.slice(0, k) == x.slice(-k); k++) {
          r[i] = x + y.slice(k);
          search(a, n, r);
        }
        r[i] = x;
      }
    }

    if(x = a[n]) {
      // there are other primes to process, so go on with the next one
      search(a, n + 1, [...r, x]);
    }
    else {
      // this is a leaf node: see if we've improved our current score
      s = r.join('');

      if(s.length <= best.length) {
        s = r.sort().join('');

        if(s.length < best.length || s < best) {
          best = s;
        }
      }
    }
  })(a);

  return best;
}

var t0 = performance.now();
for(let n = 1; n <= 25; n++) {
  console.log(`a(${n}) = ` + f(n));
  var t1 = performance.now();
  console.log("Total time: " + (t1 - t0) + " milliseconds.")
}