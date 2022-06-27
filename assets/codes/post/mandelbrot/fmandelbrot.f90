module prec
  integer, parameter :: sp = kind(1.0)
  integer, parameter :: dp = kind(1.d0)
end module prec

subroutine fMandelbrot(dat, ext, nx, ny, nmax)
  use prec
  implicit none

  ! arguments
  integer, intent(in)        :: nx, ny, nmax
  real(kind=dp), intent(in)  :: ext(4)
  real(kind=dp), intent(out), dimension(nx, ny) :: dat

  !f2py integer, optional, intent(in) :: nmax = 1000

  ! local variables
  integer          :: i, j, k
  complex(kind=dp) :: z0, z
  real(kind=dp)    :: x, y

  dat = nmax
  !$omp parallel do private(j, x, y, z0, z, k)
  do i=1, nx
    do j=1, ny
      x  = ext(1) + (ext(2) - ext(1)) / (nx - 1.) * (i - 1.)
      y  = ext(3) + (ext(4) - ext(3)) / (ny - 1.) * (j - 1.)
      z0 = dcmplx(x, y)
      z  = (0, 0)

      do k=1, nmax
        if (abs(z) > 2.0) then
          dat(i,j) = k - 1
          exit
        end if
        z = z**2 + z0
      end do
    end do
  end do
  !$omp end parallel do

  return
end subroutine fMandelbrot
