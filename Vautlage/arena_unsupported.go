//go:build !(linux && (amd64 || arm64)) && !(darwin && (amd64 || arm64))

package vaultage

import "unsafe"

// This file keeps the package compiling on platforms with no raw-mmap backend.
// Every entry point fails cleanly at arena construction, so a caller learns at
// start-up that the off-heap spine is unavailable rather than discovering it
// through undefined behaviour at run time.
//
// The engine is deliberately not given a Go-heap fallback arena. A heap-backed
// arena would be neither huge-page aligned, nor unswappable, nor invisible to
// the collector, so it would silently violate every guarantee the API makes
// while appearing to work.

func sysMapAnon(length uintptr) (unsafe.Pointer, error) {
	_ = length
	return nil, ErrUnsupportedPlatform
}

func sysMapHuge(length uintptr) (unsafe.Pointer, error) {
	_ = length
	return nil, ErrUnsupportedPlatform
}

func sysUnmap(p unsafe.Pointer, length uintptr) error {
	_, _ = p, length
	return ErrUnsupportedPlatform
}

func sysAdviseHuge(p unsafe.Pointer, length uintptr) error {
	_, _ = p, length
	return ErrUnsupportedPlatform
}

func sysPin(p unsafe.Pointer, length uintptr) error {
	_, _ = p, length
	return ErrUnsupportedPlatform
}

func sysUnpin(p unsafe.Pointer, length uintptr) error {
	_, _ = p, length
	return ErrUnsupportedPlatform
}

func sysPinThreadToCore(core int) error {
	_ = core
	return ErrUnsupportedPlatform
}
