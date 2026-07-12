/* OmniEvaluator Internal Dashboard - Minimal pub/sub store */

const _store = {
  _listeners: {},

  on(event, fn) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(fn);
  },

  off(event, fn) {
    const arr = this._listeners[event];
    if (!arr) return;
    this._listeners[event] = arr.filter((f) => f !== fn);
  },

  emit(event, detail) {
    const arr = this._listeners[event];
    if (!arr) return;
    arr.forEach((fn) => fn(detail));
  },
};
